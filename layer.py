import torch
import torch.nn as nn
import torch.distributed as dist
from parallel_state import get_tp_group, get_pp_group

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
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype = torch.float32,
        prefix='',
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.prefix = prefix

        tp_group = get_tp_group()
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        self.weight = nn.Parameter(
            torch.empty(
                (input_size, output_size // tp_size),
                dtype=params_dtype
            )
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    (output_size // tp_size,),
                    dtype=params_dtype
                )
            )
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Input tensor must have last dimension of size {self.input_size}, but got {x.shape[-1]}.")

        # Perform the column parallel matrix multiplication
        output = torch.matmul(x, self.weight)

        if self.bias is not None and not self.skip_bias_add:
            output += self.bias

        if self.gather_output:
            # Gather outputs from all TP ranks
            output_list = [torch.empty_like(output) for _ in range(self.tp_size)]
            dist.all_gather(output_list, output, group=get_tp_group())
            output = torch.cat(output_list, dim=-1)

        return output