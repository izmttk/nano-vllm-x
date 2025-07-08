import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from parallel_state import get_tp_group
from communication_op import tensor_model_parallel_all_gather

class ColumnParallelLayer(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    
    before: Y = X A + b
    after: [Y_1, ..., Y_p] = [X A_1 + b, ..., X A_p + b]

    where:

    X is the input matrix of size (batch_size, input_size),
    A is the weight matrix of size (input_size, output_size),
    b is the bias vector of size (output_size,).
    Y is the output matrix of size (batch_size, output_size).
    p is the number of GPUs, and each GPU has a portion of the weight matrix.
    A_i is the weight matrix for the i-th GPU of size (input_size, output_size / p),
    Y_i is the output matrix for the i-th GPU of size (batch_size, output_size / p).
    
    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype = torch.float32,
        tp_size: int = None,
        tp_rank: int = None
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype

        if tp_size is None:
            tp_size = get_tp_group().size()
        if tp_rank is None:
            tp_rank = get_tp_group().rank()
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank


        assert output_size % tp_size == 0, f"output_size {output_size} must be divisible by tp_size {tp_size}"
        self.output_size_per_partition = output_size // tp_size
        
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                dtype=params_dtype
            )
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    dtype=params_dtype
                )
            )
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor):
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(x, self.weight, bias)
        if self.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias