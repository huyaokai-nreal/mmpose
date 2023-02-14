import torch


class IngetralCoordinate(torch.autograd.Function):
    """Symmetry integral regression function."""
    AMPLITUDE = 2

    @staticmethod
    def forward(ctx, input):
        assert isinstance(
            input, torch.Tensor
        ), 'IngetralCoordinate only takes input as torch.Tensor'
        input_size = input.size()
        weight = torch.arange(
            input_size[-1],
            dtype=input.dtype,
            layout=input.layout,
            device=input.device)
        ctx.input_size = input_size
        output = input.mul(weight)
        ctx.save_for_backward(weight, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, output = ctx.saved_tensors
        output_coord = output.sum(dim=2, keepdim=True)
        weight = weight[None, None, :].repeat(output_coord.shape[0],
                                              output_coord.shape[1], 1)
        weight_mask = torch.ones(
            weight.shape,
            dtype=grad_output.dtype,
            layout=grad_output.layout,
            device=grad_output.device)
        weight_mask[weight < output_coord] = -1
        weight_mask[output_coord.repeat(1, 1, weight.shape[-1]) >
                    ctx.input_size[-1]] = 1
        weight_mask *= IngetralCoordinate.AMPLITUDE
        return grad_output.mul(weight_mask)
