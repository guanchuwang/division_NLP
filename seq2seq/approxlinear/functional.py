import torch
import pdb
from torch import topk
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from seq2seq.approxlinear.fdmp import FDMP


class approxmatmul_4D_only_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, A, B, scheme):
        assert len(A.shape) == 4
        assert len(B.shape) == 4

        feature_pack_B = FDMP.fdmp(B.transpose(3, 2), scheme)
        feature_pack_A = FDMP.fdmp(A, scheme)

        ctx.saved = feature_pack_B, feature_pack_A, A.shape, B.transpose(3, 2).shape
        ctx.scheme = scheme
        res = torch.matmul(A, B)
        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):

        feature_pack_B, feature_pack_A, A_shape, B_shape = ctx.saved
        scheme = ctx.scheme

        B_hat = FDMP.de_fdmp(feature_pack_B, B_shape, scheme)
        A_hat = FDMP.de_fdmp(feature_pack_A, A_shape, scheme)

        # print(B_hat.shape, A_hat.shape, B_shape, A_shape)
        # print(grad_output.shape)

        grad_A = torch.einsum('...mn, ...nd->...md', grad_output, B_hat)
        grad_B = torch.einsum('...md, ...mn-> ...dn', A_hat, grad_output)

        del feature_pack_B, feature_pack_A, B_hat, A_hat, A_shape, B_shape, scheme

        return grad_A, grad_B, None


class approxlinear_only_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias, scheme):

        feature_pack = FDMP.fdmp1d(input, scheme)
        ctx.saved = feature_pack, input.shape, weight, bias

        ctx.scheme = scheme
        res = F.linear(input, weight, bias)
        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        feature_pack, input_shape, weight, bias = ctx.saved
        scheme = ctx.scheme

        input_hat = FDMP.de_fdmp1d(feature_pack, input_shape, scheme)
        grad_weight = torch.einsum('blo, bli->oi', grad_output, input_hat)
        grad_input = torch.matmul(grad_output, weight)

        if bias is not None:
            grad_bias = grad_output.view(-1, grad_output.shape[-1]).sum(0)
        else:
            grad_bias = None

        del feature_pack, input_shape, weight, bias, input_hat, scheme

        return grad_input, grad_weight, grad_bias, None

