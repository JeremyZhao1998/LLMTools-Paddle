# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import paddle
from paddle import _C_ops
from paddle.framework import core


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm() or paddle.is_compiled_with_xpu():
        return hasattr(core.eager.ops.legacy, "fused_gemm_epilogue")
    else:
        return False


if is_fused_matmul_bias_supported():
    origin_linear = paddle.incubate.nn.functional.fused_linear
else:
    origin_linear = paddle.nn.functional.linear


class FusedLinearWithGradAdd(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        y = origin_linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        x, weight, bias = ctx.saved_tensor()
        x_grad = paddle.matmul(y_grad, weight, transpose_y=True)

        # _C_ops.fused_linear_param_grad_add(x, y_grad, dw, db, multi precision, has bias)
        if bias is None:
            if hasattr(weight, "main_grad"):
                weight.main_grad, _ = _C_ops.fused_linear_param_grad_add(
                    x, y_grad, weight.main_grad, None, True, False
                )
                return x_grad, None
            else:
                if weight.grad is not None:
                    weight.grad, _ = _C_ops.fused_linear_param_grad_add(x, y_grad, weight.grad, None, False, False)
                    return x_grad, None
                else:
                    weight_grad, _ = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False, False)
                    return x_grad, weight_grad

        if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
            weight.main_grad, bias.main_grad = _C_ops.fused_linear_param_grad_add(
                x, y_grad, weight.main_grad, bias.main_grad, True, True
            )
            return x_grad, None, None
        else:
            if weight.grad is not None:
                assert bias.grad is not None
                weight.grad, bias.grad = _C_ops.fused_linear_param_grad_add(
                    x, y_grad, weight.grad, bias.grad, False, True
                )
                return x_grad, None, None
            else:
                weight_grad, bias_grad = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False, True)
                return x_grad, weight_grad, bias_grad


def mock_layers():
    paddle.nn.functional.linear = FusedLinearWithGradAdd.apply
    if is_fused_matmul_bias_supported():
        paddle.incubate.nn.functional.fused_linear = FusedLinearWithGradAdd.apply


# register pp_reshard information to aid pp reshard
def register_pp_reshard_information(num_hidden_layers):

    from paddlenlp.trainer.utils.reshard.pp_reshard import (
        register_index_layer_func,
        register_layername_prefix,
        regitser_extract_layer_name_func,
    )

    # register layer names
    register_layername_prefix("column_sequence_parallel_linear")
    register_layername_prefix("row_sequence_parallel_linear")
    register_layername_prefix("linear")
    register_layername_prefix("embedding")
    register_layername_prefix("create_parameter")
    register_layername_prefix("llama_lm_head")

    # register func to extract layer from stuctural param name
    # register func to extract layer index  from stuctural param name

    def extract_layer_name(param_name):
        patterns = [r"^llama\.embed_tokens", "^llama\.norm", r"^lm_head", r"^llama\.layers((\.\d+))"]
        # match 1
        for p in patterns:
            match = re.search(p, param_name)
            if match:
                return match.group()

    def index_layer(layer_name):
        if layer_name == "llama.embed_tokens":
            return 0
        elif layer_name == "llama.norm":
            return num_hidden_layers + 1
        elif layer_name == "lm_head":
            return num_hidden_layers + 2
        else:
            pattern = r"llama\.layers((\.(\d+)))"
            match = re.search(pattern, layer_name)
            assert match
            index = int(match.group(3)) + 1
            assert index <= num_hidden_layers, f"{index} {num_hidden_layers}"
            return index

    regitser_extract_layer_name_func(extract_layer_name)
    register_index_layer_func(index_layer)
