# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, missing-module-docstring
# pylint: disable=unused-argument, invalid-name
from functools import reduce
import operator

from . import cuda
from .._lib import register_compute
from .._lib import generic_func
from .._lib import tvm as _tvm
from .._lib import _reg
from .._lib import strategy
from .._lib import random

_topi = _tvm.topi  # pylint: disable=no-member

_reg.register_injective_schedule("raf.op.tvm.pad")

_reg.register_strategy("raf.op.tvm.dense", strategy.dense_strategy)


def compute_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=False):
    if len(inputs) == 2:
        data, weight = inputs[0], inputs[1]
    else:
        raise ValueError("Invalid input")
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    return [_topi.matmul(data, weight, transp_a=transpose_a, transp_b=transpose_b)]


@register_compute("raf.op.tvm.matmul")
def compute_matmul(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=False)


@register_compute("raf.op.tvm.matmul_tn")
def compute_matmul_tn(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=True, transpose_b=False)


@register_compute("raf.op.tvm.matmul_nt")
def compute_matmul_nt(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=True)


@register_compute("raf.op.tvm.matmul_tt")
def compute_matmul_tt(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=True, transpose_b=True)


_reg.register_injective_schedule("raf.op.tvm.matmul")
_reg.register_injective_schedule("raf.op.tvm.matmul_tn")
_reg.register_injective_schedule("raf.op.tvm.matmul_nt")
_reg.register_injective_schedule("raf.op.tvm.matmul_tt")


def compute_batch_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=False):
    assert len(inputs) == 2, "Expected 2 inputs, but got {}".format(len(inputs))
    data, weight = inputs[0], inputs[1]
    assert len(data.shape) == 3 and len(weight.shape) == 3, "only support 3-dim batch matmul"

    # Topi batch matmul currently support NT mode. So, add transposes when it is not NT
    if transpose_a:
        data = _topi.transpose(data, (0, 2, 1))
    if not transpose_b:
        weight = _topi.transpose(weight, (0, 2, 1))
    return [_topi.nn.batch_matmul(data, weight)]


@register_compute("raf.op.tvm.batch_matmul")
def compute_batch_matmul_nn(attr, inputs, output_type):
    return compute_batch_matmul_general(
        attr, inputs, output_type, transpose_a=False, transpose_b=False
    )


@register_compute("raf.op.tvm.batch_matmul_tn")
def compute_batch_matmul_tn(attr, inputs, output_type):
    return compute_batch_matmul_general(
        attr, inputs, output_type, transpose_a=True, transpose_b=False
    )


@register_compute("raf.op.tvm.batch_matmul_tt")
def compute_batch_matmul_tt(attr, inputs, output_type):
    return compute_batch_matmul_general(
        attr, inputs, output_type, transpose_a=True, transpose_b=True
    )


_reg.register_injective_schedule("raf.op.tvm.batch_matmul")
_reg.register_injective_schedule("raf.op.tvm.batch_matmul_tn")
_reg.register_injective_schedule("raf.op.tvm.batch_matmul_tt")

_reg.register_strategy("raf.op.tvm.batch_matmul_nt", strategy.batch_matmul_strategy)

_reg.register_strategy("raf.op.tvm.softmax", strategy.softmax_strategy)


@register_compute("raf.op.tvm.softmax_dx")
def compute_softmax_dx(attr, inputs, output_type):
    y, dy = inputs[0], inputs[1]
    axis = attr.axis
    return [(dy - _topi.sum(dy * y, axis, True)) * y]


@generic_func
def schedule_softmax_dx(attrs, outs, target):
    # FIXME: softmax_dx is not an injective op so we should not use inject schedule.
    with target:
        return _topi.generic.schedule_injective(outs)


@schedule_softmax_dx.register(["cuda", "gpu"])
def schedule_softmax_dx_cuda(attrs, outs, _):
    mul2 = outs[0]
    sub, _ = mul2.op.input_tensors  # input_tensors[1] = dy
    _, red = sub.op.input_tensors  # input_tensors[0] = y
    mul1 = red.op.input_tensors[0]

    axis = attrs.axis
    ndim = len(mul2.shape)
    axis = int(axis) if axis is not None else ndim - 1
    if axis >= ndim:
        axis %= ndim
    while axis < 0:
        axis += ndim

    sch = _tvm.te.create_schedule([mul2.op])
    thd_x = _tvm.te.thread_axis("threadIdx.x")
    blk_x = _tvm.te.thread_axis("blockIdx.x")

    sch[sub].compute_inline()
    sch[mul1].compute_inline()
    _, mul2_r_i = sch[mul2].split(mul2.op.axis[axis], factor=32)
    sch[mul2].bind(mul2_r_i, thd_x)

    _, red_r_i = sch[red].split(red.op.reduce_axis[0], factor=32)
    sch[red].bind(red_r_i, thd_x)

    # This is a more aggressive optimzation that only works for limited workloads.
    if ndim > 1 and axis in [0, ndim - 1]:
        sch[red].compute_at(sch[mul2], mul2.op.axis[axis - 1])
        fused = sch[mul2].fuse(*mul2.op.axis[0:axis])
        sch[mul2].bind(fused, blk_x)

    sch[red].pragma(red.op.axis[0], "auto_unroll_max_step", 64)
    sch[red].pragma(red.op.axis[0], "unroll_explicit", True)
    return sch


_reg.register_schedule("raf.op.tvm.softmax_dx", schedule_softmax_dx)

_reg.register_schedule("raf.op.tvm.avg_pool2d", strategy.schedule_pool)
# TODO(@XIAO-XIA): cuda schedule should be complemented after the implementation of auto schedule

_reg.register_schedule("raf.op.tvm.avg_pool2d_dx", strategy.schedule_pool_grad)

_reg.register_schedule("raf.op.tvm.max_pool2d", strategy.schedule_pool)
# TODO(@XIAO-XIA): cuda schedule should be complemented after the implementation of auto schedule

_reg.register_schedule("raf.op.tvm.max_pool2d_dx", strategy.schedule_pool_grad)

_reg.register_schedule("raf.op.tvm.adaptive_avg_pool2d", strategy.schedule_adaptive_pool)
_reg.register_schedule("raf.op.tvm.adaptive_avg_pool2d_dx", strategy.schedule_pool_grad)
_reg.register_schedule("raf.op.tvm.adaptive_max_pool2d", strategy.schedule_adaptive_pool)
_reg.register_schedule("raf.op.tvm.adaptive_max_pool2d_dx", strategy.schedule_pool_grad)


@generic_func
def schedule_log_softmax(attrs, outs, target):
    # Use the TVM schedules for other targets.
    with target:
        return _topi.generic.schedule_softmax(outs)


@schedule_log_softmax.register(["cuda", "gpu"])
def schedule_log_softmax_cuda(attrs, outs, _):
    """Override the CUDA schedule for better performance and fusion support."""
    out = outs[0]

    axis = attrs.axis
    ndim = len(out.shape)
    assert ndim == 2, "Only support 2-D log_softmax"
    axis = int(axis) if axis is not None else 1
    if axis >= ndim:
        axis %= ndim
    while axis < 0:
        axis += ndim

    sch = _tvm.te.create_schedule([out.op])
    thd_x = _tvm.te.thread_axis("threadIdx.x")
    blk_x = _tvm.te.thread_axis("blockIdx.x")

    inp, maxelem, expsum = out.op.input_tensors

    (out_local,) = sch.cache_write([out], "local")
    sch[out_local].compute_inline()

    _, out_j_i = sch[out].split(out.op.axis[1], factor=32)
    sch[out].bind(out_j_i, thd_x)

    _, maxelem_k_i = sch[maxelem].split(maxelem.op.reduce_axis[0], factor=32)
    sch[maxelem].bind(maxelem_k_i, thd_x)
    sch[maxelem].compute_at(sch[out], out.op.axis[0])

    _, expsum_k_i = sch[expsum].split(expsum.op.reduce_axis[0], factor=32)
    sch[expsum].bind(expsum_k_i, thd_x)
    sch[expsum].compute_at(sch[out], out.op.axis[0])

    sch[out].bind(out.op.axis[0], blk_x)
    sch[maxelem].pragma(maxelem.op.axis[0], "auto_unroll_max_step", 64)
    sch[maxelem].pragma(maxelem.op.axis[0], "unroll_explicit", True)
    sch[expsum].pragma(expsum.op.axis[0], "auto_unroll_max_step", 64)
    sch[expsum].pragma(expsum.op.axis[0], "unroll_explicit", True)

    # If input is fused with another op, then try to inline it.
    # In case the fused input op cannot be inlined (e.g., not elementwise),
    # this function simply throw exception and let the dispatcher handle it.
    if isinstance(inp.op, _tvm.te.tensor.ComputeOp):
        sch[inp].compute_inline()
    return sch


_reg.register_schedule("raf.op.tvm.log_softmax", schedule_log_softmax)


@register_compute("raf.op.tvm.log_softmax_dx")
def compute_log_softmax_dx(attr, inputs, output_type):
    # The grad function of log_softmax decomposes log_softmax_dx to a series of RAF IR ops
    # so this function is not used. It only kept in case we want to have a powerful schedule
    # especially for this op in the future.
    y, dy = inputs[0], inputs[1]
    axis = attr.axis
    return [dy - _topi.exp(y) * _topi.sum(dy, axis, False)]


_reg.register_injective_schedule("raf.op.tvm.log_softmax_dx")


@register_compute("raf.op.tvm._contrib_dropout")
def compute_contrib_dropout(attr, inputs, output_type):
    # pylint: disable=import-outside-toplevel
    x = inputs[0]
    p = attr.rate
    if x.dtype != "float32" and x.dtype != "float64":
        raise TypeError(
            "input array of raf.dropout is expected to be the type of float32 "
            + "or float64, but received {}".format(x.dtype)
        )
    if p < 0.0 or p >= 1:
        raise ValueError("p is out of interval")
    retain_p = _tvm.tir.const(1 - p, x.dtype)
    mask = random.uniform(0, 1, x.shape)
    ret = _tvm.te.compute(
        x.shape,
        lambda *ix: _tvm.te.if_then_else(
            mask[ix] <= _tvm.tir.const(p, "float32"), _tvm.tir.const(0, x.dtype), x[ix] / retain_p
        ),
    )
    mask = _tvm.te.compute(
        x.shape,
        lambda *ix: _tvm.te.if_then_else(
            mask[ix] <= _tvm.tir.const(p, "float32"),
            _tvm.tir.const(0, "float32"),
            _tvm.tir.const(1 / (1 - p), "float32"),
        ),
    )
    # states and reserve_space are valid in cudnn only
    states = _topi.full((), dtype="uint8", fill_value=0.0)
    reserve_space_shape = ()
    if len(output_type.fields[-1].shape) > 0:
        # Reserve_space is not scalar type. It is dispatched from the base op
        from .._ffi.backend.cudnn import GetDropoutReserveSpaceSizeInBytes

        if GetDropoutReserveSpaceSizeInBytes:
            x_ty = _tvm.relay.TensorType(x.shape, dtype=x.dtype)
            reserve_space_shape = (GetDropoutReserveSpaceSizeInBytes(x_ty),)
    reserve_space = _topi.full(reserve_space_shape, dtype="uint8", fill_value=0.0)
    return [ret, mask, states, reserve_space]


_reg.register_injective_schedule("raf.op.tvm._contrib_dropout")


@register_compute("raf.op.tvm._contrib_dropout_dx")
def compute_contrib_dropout_dx(attr, inputs, output_type):
    dy = inputs[0]
    mask = inputs[1]
    assert _topi.utils.get_const_tuple(dy.shape) == _topi.utils.get_const_tuple(
        mask.shape
    ), "dy.shape %s != mask.shape %s" % (str(dy.shape), str(mask.shape))
    ret = _tvm.te.compute(dy.shape, lambda *idx: dy[idx] * _tvm.topi.cast(mask[idx], dy.dtype))
    return [ret]


_reg.register_injective_schedule("raf.op.tvm._contrib_dropout_dx")


@register_compute("raf.op.tvm.relu_dx")
@_tvm.te.tag_scope(tag=_tvm.topi.tag.ELEMWISE)
def compute_relu_dx(attr, inputs, output_type):
    grad_mode = attr.grad_mode
    if grad_mode == "both":
        data, dy = inputs[0], inputs[2]
    else:
        data, dy = inputs[0], inputs[1]
    # For y = relu(x), x or y can be used to calcluate graident
    # if both x and y are given, we use x here
    # Using x: return 0 if x < 0 else dy
    # Using y: return 0 if y == 0 else dy
    G = _tvm.te.compute(
        dy.shape,
        lambda *idx: _tvm.te.if_then_else(data[idx] <= 0, _tvm.tir.const(0, dy.dtype), dy[idx]),
    )
    return [G]


_reg.register_injective_schedule("raf.op.tvm.relu_dx")


@register_compute("raf.op.tvm.threshold")
def compute_threshold(attr, inputs, output_type):
    x = inputs[0]
    threshold = _tvm.tir.const(attr.threshold, x.dtype)
    value = _tvm.tir.const(attr.value, x.dtype)
    return [
        _tvm.te.compute(
            x.shape,
            lambda *idx: _tvm.te.if_then_else(x[idx] > threshold, x[idx], value),
            tag=_tvm.topi.tag.ELEMWISE,
        )
    ]


_reg.register_injective_schedule("raf.op.tvm.threshold")


@register_compute("raf.op.tvm.threshold_dx")
def compute_threshold_dx(attr, inputs, output_type):
    x, dy = inputs[0], inputs[1]
    threshold = _tvm.tir.const(attr.threshold, x.dtype)
    return [
        _tvm.te.compute(
            dy.shape,
            lambda *idx: _tvm.te.if_then_else(
                x[idx] > threshold, dy[idx], _tvm.tir.const(0, dy.dtype)
            ),
            tag=_tvm.topi.tag.ELEMWISE,
        )
    ]


_reg.register_injective_schedule("raf.op.tvm.threshold_dx")


@register_compute("raf.op.tvm.layer_norm")
def compute_layer_norm(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    set_scale = attr.set_scale_bias
    x = inputs[0]
    if set_scale:
        scale = inputs[1]
        bias = inputs[2]
    axis, eps = _topi.utils.get_const_int(attr.axis), _tvm.tir.const(attr.epsilon, dtype=x.dtype)
    ndim = len(x.shape)
    if axis < 0:
        axis = ndim + axis

    def pad(data, target):
        newaxis = []
        for i in range(ndim):
            if i != axis:
                newaxis.append(i)
        return _topi.expand_like(data, target, newaxis)

    count = _tvm.tir.const(1, dtype=x.dtype)
    count *= x.shape[axis]
    reduce_axes = [axis]
    x_sum = _topi.sum(x, reduce_axes, keepdims=True)
    x_mean = _topi.divide(x_sum, count)
    sq_diff = _topi.power(_topi.subtract(x, x_mean), 2)
    sq_diff_sum = _topi.sum(sq_diff, reduce_axes, keepdims=True)
    x_var = _topi.divide(sq_diff_sum, count)
    denominator = _topi.sqrt(_topi.add(x_var, eps))
    out = _topi.divide(_topi.subtract(x, x_mean), denominator)
    if set_scale:
        pscale = pad(scale, out)
        out = _topi.multiply(pscale, out)
        out = _topi.add(out, pad(bias, out))
    return [out]


@generic_func
def schedule_generic(attrs, outs, target):
    with target:
        return _topi.generic.schedule_injective(outs)


@schedule_generic.register(["cuda", "gpu"])
def schedule_generic_cuda(attrs, outs, target):
    with target:
        out = outs[0]
        s = cuda.injective.schedule_injective(outs)
        # fuse axes and split into bx and tx then bind
        scheduled_ops = []
        num_thread = 64

        def bind_axes(s, out):
            if (
                isinstance(out.op, _tvm.te.ComputeOp)
                and isinstance(out.op.body[0], _tvm.tir.expr.Reduce)
                and len(s[out].iter_var_attrs) == 0
                and out.op not in scheduled_ops
            ):
                scheduled_ops.append(out.op)
                fused = s[out].fuse(*s[out].op.axis)
                bx, tx = s[out].split(fused, factor=num_thread)
                s[out].bind(bx, _tvm.te.thread_axis("blockIdx.x"))
                s[out].bind(tx, _tvm.te.thread_axis("threadIdx.x"))
            for inp in out.op.input_tensors:
                bind_axes(s, inp)

        bind_axes(s, out)
        return s


_reg.register_schedule("raf.op.tvm.layer_norm", schedule_generic)


@register_compute("raf.op.tvm.layer_norm_dx")
def compute_layer_norm_dx(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    set_scale = attr.set_scale_bias
    if set_scale:
        x, scale, dy = inputs
    else:
        x, dy = inputs[0], inputs[1]
    axis, eps = _topi.utils.get_const_int(attr.axis), _tvm.tir.const(attr.epsilon, dtype=x.dtype)
    ndim = len(x.shape)
    if axis < 0:
        axis = ndim + axis

    count = x.shape[axis]
    reduce_axes = [axis]
    x_sum = _topi.sum(x, reduce_axes, keepdims=True)
    x_mean = _topi.divide(x_sum, count)
    sq_diff = _topi.power(_topi.subtract(x, x_mean), 2)
    sq_diff_sum = _topi.sum(sq_diff, reduce_axes, keepdims=True)
    x_var = _topi.divide(sq_diff_sum, count)
    denominator = _topi.sqrt(_topi.add(x_var, eps))
    xmu = _topi.subtract(x, x_mean)

    bar_x = _topi.divide(xmu, denominator)
    w = _topi.divide(dy, denominator)

    def pad(data, target):
        newaxis = []
        for i in range(ndim):
            if i != axis:
                newaxis.append(i)
        return _topi.expand_like(data, target, newaxis)

    if set_scale:
        w = w * pad(scale, w)
    w_sum = _topi.sum(w, reduce_axes, keepdims=True)
    mean_w = _topi.divide(w_sum, count)
    w_times_bar_x = _topi.multiply(w, bar_x)
    w_times_bar_x_sum = _topi.sum(w_times_bar_x, reduce_axes, keepdims=True)
    mean_w_times_bar_x = _topi.divide(w_times_bar_x_sum, count)
    dx = _topi.subtract(w, mean_w)
    dx = _topi.subtract(dx, _topi.multiply(bar_x, mean_w_times_bar_x))
    if set_scale:
        reduce_axes = list(range(axis)) + list(range(axis + 1, ndim))
        dw = _topi.sum(dy * (x - x_mean) / denominator, axis=reduce_axes)
        db = _topi.sum(dy, axis=reduce_axes)
        return [dx, dw, db]
    return [dx]


_reg.register_schedule("raf.op.tvm.layer_norm_dx", schedule_generic)

_reg.register_strategy("raf.op.tvm.conv2d", strategy.conv2d_strategy)

_reg.register_strategy("raf.op.tvm.conv2d_transpose", strategy.conv2d_transpose_strategy)


def declaration_conv2d_transpose_impl(data, kernel, strides, padding, out_dtype, output_padding):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    """Implementation of conv2d transpose"""
    data_pad, kernel_transform = _topi.nn.conv2d_transpose_nchw_preprocess(
        data, kernel, strides, padding, out_dtype, (0, 0)
    )
    batch, in_c, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape

    # convolution stage
    out_c = _topi.nn.simplify(out_c)
    out_h = _topi.nn.simplify(in_h - filter_h + 1 + output_padding[0])
    out_w = _topi.nn.simplify(in_w - filter_w + 1 + output_padding[1])
    dc = _tvm.te.reduce_axis((0, in_c), name="dc")
    dh = _tvm.te.reduce_axis((0, filter_h), name="dh")
    dw = _tvm.te.reduce_axis((0, filter_w), name="dw")
    Output = _tvm.te.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: _tvm.tir.sum(
            data_pad[b, dc, h + dh, w + dw].astype(out_dtype)
            * kernel_transform[c, dc, dh, dw].astype(out_dtype),
            axis=[dc, dh, dw],
        ),
        tag="conv2d_transpose_nchw",
    )
    return Output


@register_compute("raf.op.tvm.conv2d_dx")
def compute_conv2d_dx(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, dilation, layout = (
        _topi.utils.get_const_tuple(attr.strides),
        _topi.utils.get_const_tuple(attr.padding),
        _topi.utils.get_const_tuple(attr.dilation),
        attr.data_layout,
    )
    assert layout == "NCHW"
    assert dilation == (1, 1), "dilation is not supported yet"
    assert attr.groups == 1, "only support groups = 1"
    use_output = attr.use_output
    if use_output:
        W, dy = inputs[0], inputs[2]
    else:
        W, dy = inputs[0], inputs[1]
    X = _tvm.te.placeholder(shape=attr.kernel_size, dtype=dy.dtype)
    R = _topi.nn.conv2d(X, W, strides, padding, dilation, layout)
    grads = _tvm.te.gradient(R, [X], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.conv2d_dx", schedule_generic)


@register_compute("raf.op.tvm.conv2d_dw")
def compute_conv2d_dw(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, dilation, layout = (
        _topi.utils.get_const_tuple(attr.strides),
        _topi.utils.get_const_tuple(attr.padding),
        _topi.utils.get_const_tuple(attr.dilation),
        attr.data_layout,
    )
    assert layout == "NCHW", "only support NCHW layout"
    assert dilation == (1, 1), "dilation is not supported yet"
    assert attr.groups == 1, "only support groups = 1"

    use_output = attr.use_output
    if use_output:
        X, dy = inputs[0], inputs[2]
    else:
        X, dy = inputs[0], inputs[1]

    W = _tvm.te.placeholder(shape=attr.kernel_size, dtype=X.dtype)
    R = _topi.nn.conv2d(X, W, strides, padding, dilation, layout)
    grads = _tvm.te.gradient(R, [W], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.conv2d_dw", schedule_generic)


@register_compute("raf.op.tvm.conv2d_transpose_dx")
def compute_conv2d_transpose_dx(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, output_padding, dilation, layout = (
        _topi.utils.get_const_tuple(attr.strides),
        _topi.utils.get_const_tuple(attr.padding),
        _topi.utils.get_const_tuple(attr.output_padding),
        _topi.utils.get_const_tuple(attr.dilation),
        attr.data_layout,
    )
    assert layout == "NCHW", "only support NCHW layout"
    assert dilation == (1, 1), "dilation is not supported yet"
    assert attr.groups == 1, "only support groups = 1"
    use_output = attr.use_output
    if use_output:
        W, dy = inputs[0], inputs[2]
    else:
        W, dy = inputs[0], inputs[1]
    assert (
        W.shape[3] > 1 and W.shape[2] > 1
    ), "not support kernel size 1 for now. \
                                                See apache/tvm#8087"
    X = _tvm.te.placeholder(shape=attr.kernel_size, dtype=dy.dtype)
    R = _topi.x86.conv2d_transpose_nchw(X, W, strides, padding, dy.dtype, output_padding)
    grads = _tvm.te.gradient(R, [X], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.conv2d_transpose_dx", schedule_generic)


@register_compute("raf.op.tvm.conv2d_transpose_dw")
def compute_conv2d_transpose_dw(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, output_padding, dilation, layout = (
        _topi.utils.get_const_tuple(attr.strides),
        _topi.utils.get_const_tuple(attr.padding),
        _topi.utils.get_const_tuple(attr.output_padding),
        _topi.utils.get_const_tuple(attr.dilation),
        attr.data_layout,
    )
    assert layout == "NCHW", "only support NCHW layout"
    assert dilation == (1, 1), "dilation is not supported yet"
    assert attr.groups == 1, "only support groups = 1"

    use_output = attr.use_output
    if use_output:
        X, dy = inputs[0], inputs[2]
    else:
        X, dy = inputs[0], inputs[1]

    W = _tvm.te.placeholder(shape=attr.kernel_size, dtype=X.dtype)
    R = _topi.x86.conv2d_transpose_nchw(X, W, strides, padding, dy.dtype, output_padding)

    grads = _tvm.te.gradient(R, [W], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.conv2d_transpose_dw", schedule_generic)


def average(data, axis):
    shape = _topi.utils.get_const_tuple(data.shape)
    shape = [shape[i] for i in axis]
    size = reduce(operator.mul, shape, 1)
    tot = _topi.sum(data, axis=axis)
    return _topi.divide(tot, size)


@register_compute("raf.op.tvm.batch_norm_train")
def batch_norm_train_compute(attrs, inputs, output_type):  # pylint: disable=too-many-locals
    x, running_m0, running_v0, w, b = inputs
    momentum, eps = attrs.momentum, attrs.eps
    shape = _topi.utils.get_const_tuple(x.shape)
    ndim = len(shape)
    axis = 1
    num_newaxis = ndim - axis - 1
    reduce_axes = list(range(axis)) + list(range(axis + 1, ndim))
    reduce_shape = [shape[i] for i in reduce_axes]
    reduce_size = reduce(operator.mul, reduce_shape, 1)

    def pad(data):
        return _topi.expand_dims(data, axis=1, num_newaxis=num_newaxis)

    mean = average(x, axis=reduce_axes)
    x_sq = _topi.multiply(x, x)
    sq_mean = average(x_sq, axis=reduce_axes)
    mean_sq = _topi.multiply(mean, mean)
    var = sq_mean - mean_sq
    running_m = running_m0 * (1 - momentum) + mean * momentum
    running_v = running_v0 * (1 - momentum) + var * reduce_size / (reduce_size - 1) * momentum
    var_add_eps = _topi.add(var, eps)
    sqrt_var = _topi.sqrt(var_add_eps)
    scale = _topi.divide(w, sqrt_var)
    neg_mean = _topi.negative(mean)
    shift = _topi.multiply(neg_mean, scale)
    shift = _topi.add(shift, b)
    y = _topi.add(_topi.multiply(x, pad(scale)), pad(shift))
    return [y, running_m, running_v]


_reg.register_reduce_schedule("raf.op.tvm.batch_norm_train")


@register_compute("raf.op.tvm.batch_norm_infer")
def batch_norm_infer_compute(attrs, inputs, output_type):  # pylint: disable=too-many-locals
    x, running_m, running_v, w, b = inputs
    eps = attrs.eps
    shape = _topi.utils.get_const_tuple(x.shape)
    ndim = len(shape)
    axis = 1
    num_newaxis = ndim - axis - 1

    def pad(data):
        return _topi.expand_dims(data, axis=1, num_newaxis=num_newaxis)

    var_add_eps = _topi.add(running_v, eps)
    sqrt_var = _topi.sqrt(var_add_eps)
    scale = _topi.divide(w, sqrt_var)
    neg_mean = _topi.negative(running_m)
    shift = _topi.multiply(neg_mean, scale)
    shift = _topi.add(shift, b)
    y = _topi.add(_topi.multiply(x, pad(scale)), pad(shift))
    return [y]


_reg.register_injective_schedule("raf.op.tvm.batch_norm_infer")


@register_compute("raf.op.tvm.batch_norm_train_dxwb")
def batch_norm_train_dxwb_compute(attrs, inputs, output_type):  # pylint: disable=too-many-locals
    dy, x, w, _ = inputs
    eps = attrs.eps
    shape = _topi.utils.get_const_tuple(x.shape)
    ndim = len(shape)
    axis = 1
    num_newaxis = ndim - axis - 1
    reduce_axes = list(range(axis)) + list(range(axis + 1, ndim))
    reduce_shape = [shape[i] for i in reduce_axes]
    reduce_size = reduce(operator.mul, reduce_shape, 1)

    def pad(data):
        return _topi.expand_dims(data, axis=1, num_newaxis=num_newaxis)

    mean = average(x, axis=reduce_axes)
    x_sq = _topi.multiply(x, x)
    sq_mean = average(x_sq, axis=reduce_axes)
    mean_sq = _topi.multiply(mean, mean)
    var = sq_mean - mean_sq
    inv_sqrt_var = 1 / _topi.sqrt(var + eps)
    sum_dy_x = _topi.sum(dy * x, axis=reduce_axes)
    sum_dy = _topi.sum(dy, axis=reduce_axes)
    db = sum_dy
    dw = (sum_dy_x - mean * sum_dy) * inv_sqrt_var
    dx = (
        dy - pad(db / reduce_size) - (x - pad(mean)) * pad(dw * inv_sqrt_var) / reduce_size
    ) * pad(w * inv_sqrt_var)
    return [dx, dw, db]


_reg.register_reduce_schedule("raf.op.tvm.batch_norm_train_dxwb")
