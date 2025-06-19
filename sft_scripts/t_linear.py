import os
import torch
import habana_frameworks.torch.core as htcore
from torch.nn import Module
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.autograd import Function

class FP4LinearF(Function):
    @staticmethod
    def forward(
        ctx,
        input:torch.Tensor,
        weight:torch.Tensor,
        bias:Optional[torch.Tensor] = None,
        first_call:bool=True,
    ) -> torch.Tensor:

        if bias is not None:
            out = input @ weight.T + bias
        else:
            out = input @ weight.T

        ctx.save_for_backward(
            input,
            weight,
            bias,
        )
        ctx.first_call = first_call

        return out


    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            import pydevd
            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        (
            input,
            weight,
            bias
        ) = ctx.saved_tensors

        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ input
        print(grad_weight)
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)
            print(grad_bias)
            return grad_input, grad_weight, grad_bias, None
        else:
            return grad_input, grad_weight, None, None


class FP4Linear(Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_call = True

    def forward(self,
                input:torch.Tensor,
                weight:torch.Tensor,
                bias:Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # assert bias is None # NOTE: In llama2-7b bias is None.
        # print_rank_0(f"=********************================bias: {bias}")
        # exit()

        A, B, C = input.shape
        out = FP4LinearF.apply(input.reshape(A * B, C), weight, bias, self.first_call)
        self.first_call = False
        return out.reshape(A, B, -1)

a = torch.randn([1, 1, 3], requires_grad=True)
weight = torch.randn([2,3], requires_grad=True)
bias = torch.randn([2], requires_grad=True)

a_2 = a.clone()
weight_2 = weight.clone()
bias_2 = bias.clone()


# out_1 = F.linear(a.reshape(1, 3), weight, None)
out_1 = F.linear(a, weight, bias)

y_1 = out_1.mean()
y_1.backward()

print(out_1)
print(y_1)
print(weight.grad)
print(bias.grad)


# exit()
new_linear = FP4Linear()
out_2 = new_linear(a_2, weight_2, bias_2)
print(out_2)
y_2 = out_2.mean()
print(y_2)
y_2.backward()

#print(weight_2.grad)
#print(bias_2.grad)
