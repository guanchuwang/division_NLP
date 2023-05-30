import torch
import pdb
from torch import topk
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd


class approxmatmul_4D_fw_and_bw(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, A, B, ratio, minimal_k, scheme, batch_dim_use_same_indices=True):
        assert len(A.shape) == 4
        assert len(B.shape) == 4
        if scheme is not None:
            grad_A, grad_B = scheme.get_scale()
        else:
            grad_A, grad_B = None, None

        if ratio == 1.0 or grad_A is None or grad_B is None:
            B_top_k = B
            A_top_k = A
            top_k_indices = None
        else:
            A_top_k, B_top_k, top_k_indices = weighted_subsample_4D_AB(A, B, grad_B, ratio, minimal_k,
                                                        batch_dim_use_same_indices=batch_dim_use_same_indices)

        ctx.saved = A_top_k, B_top_k, top_k_indices, A.shape, B.shape
        ctx.scheme = scheme
        res = torch.matmul(A_top_k, B_top_k)
        return res


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        A_top_k, B_top_k, top_k_indices, A_shape, B_shape = ctx.saved
        if top_k_indices is None: # B_top_k.shape[3] == B_shape[3]:
            grad_A = torch.einsum('...mn, ...dn-> ...md', grad_output, B_top_k)
            grad_B  = torch.einsum('...md, ...mn-> ...dn', A_top_k, grad_output)
        else:
            grad_A = torch.zeros(A_shape, device=A_top_k.device)
            grad_A_ = torch.einsum('...mn, ...kn-> ...mk', grad_output, B_top_k)
            grad_A = grad_A.scatter_(3, index=top_k_indices.unsqueeze(2).expand(*A_shape[:3], top_k_indices.shape[-1]), src=grad_A_)


            grad_B = torch.zeros(B_shape, device=B_top_k.device)
            grad_B_  = torch.einsum('...mk, ...mn-> ...kn', A_top_k, grad_output)
            grad_B = grad_B.scatter_(2, index=top_k_indices.unsqueeze(-1).expand(*top_k_indices.shape, grad_B.shape[-1]), src=grad_B_)
            # torch.arange(b*h).unsqueeze(1)

        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)
        return grad_A, grad_B, None, None, None, None


class approxmatmul_4D_only_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, A, B, ratio, deter_ratio, minimal_k, scheme, batch_dim_use_same_indices=True):
        assert len(A.shape) == 4
        assert len(B.shape) == 4
        if scheme is not None:
            grad_A, grad_B = scheme.get_scale()
        else:
            grad_A, grad_B = None, None

        if ratio == 1.0 or grad_A is None or grad_B is None:
            ctx.saved = B.transpose(3, 2), None, None, None, None, A, None, None, None, None, ratio, A.shape, B.shape
        else:
            B_top_k_trans, top_k_indices_B = subsample_4D_B_by_norm(grad_A, B.transpose(3, 2), None, ratio, deter_ratio,
                                                                    minimal_k, batch_dim_use_same_indices)

            A_top_k, top_k_indices_A = subsample_4D_B_by_norm(grad_B, A, None, ratio, deter_ratio,
                                                              minimal_k, batch_dim_use_same_indices)

            ctx.saved = B_top_k_trans, top_k_indices_B, A_top_k, top_k_indices_A, ratio, A.shape, B.shape

        ctx.scheme = scheme
        res = torch.matmul(A, B)
        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):

        B_top_k_trans, top_k_indices_B, A_top_k, top_k_indices_A, ratio, A_shape, B_shape = ctx.saved

        # b, h, m, d = A_shape
        # _, _, _, n = B_shape
        # _, _, _, k = B_top_k.shape
        # print("B={}, h={}, m={}, n={}, d={}, k={}".format(b, h, m, n, d, k))
        # print(A_shape, B_shape)
        # print(A_top_k.shape, B_top_k.shape)
        if top_k_indices_B is None: # B_top_k.shape[3] == B_shape[3]:
            grad_A = torch.einsum('...mn, ...nd->...md', grad_output, B_top_k_trans)
        else:
            if top_k_indices_B.ndim == 1:
                grad_A = torch.einsum('bhmk, bhkd->bhmd', grad_output[..., top_k_indices_B], B_top_k_trans)

            else:
                # grad_output: b x h x m x n
                # top_k_indices_B: bh x k
                # top_k_indices_gather_B: b x h x m x k
                # grad_output_ Shape: b x h x m x k
                # grad_A Shape: b x h x m x d
                grad_output_ = torch.gather(grad_output, dim=3, index=top_k_indices_B.unsqueeze(2).expand(*A_shape[:3], top_k_indices_B.shape[-1]))
                grad_A = torch.einsum('...mk, ...kd->...md', grad_output_, B_top_k_trans)

        if top_k_indices_A is None: # A_top_k.shape[2] == A_shape[2]:
            grad_B = torch.einsum('...md, ...mn-> ...dn', A_top_k, grad_output)
        else:
            if top_k_indices_A.ndim == 1:
                grad_B = torch.einsum('bhnk, bhkd->bhnd', grad_output.transpose(3, 2)[..., top_k_indices_A], A_top_k).transpose(3, 2)
            else:
                # grad_output: b x h x m x n
                # top_k_indices_A: bh x k
                # top_k_indices_gather_A: b x h x k x n
                # grad_output_: b x h x k x n
                # grad_B Shape: b x h x d x n
                grad_output_ = torch.gather(grad_output, dim=2, index=top_k_indices_A.unsqueeze(-1).expand(*top_k_indices_A.shape, grad_output.shape[-1]))
                grad_B = torch.einsum('...kd, ...kn-> ...dn', A_top_k, grad_output_)

        if ctx.scheme:
            # print("A Shape:", A_shape, "B Shape:", B_shape)
            ctx.scheme.set_scale(grad_output)

        del B_top_k_trans, top_k_indices_B, A_top_k, top_k_indices_A, ratio, A_shape, B_shape

        return grad_A, grad_B, None, None, None, None, None


@torch.no_grad()
def subsample_4D_B_by_norm(A, B, weight, sample_ratio, deter_ratio, minimal_k, batch_dim_use_same_indices: bool = True):
    # A: b x h x n
    # B: b x h x n x d
    # print(f'A: {A.shape}')
    # print(f'B: {B.shape}')
    b, h, _ = A.shape
    _, _, n, d = B.shape
    k_candidate = int(n * sample_ratio)
    k = min(max(k_candidate, minimal_k), n)

    if k == n:
        return B, None

    if batch_dim_use_same_indices:
        A_ = A.reshape(-1, A.shape[-1])  # bh x n
        B = B.transpose(3, 2)  # b x h x d x n
        B_ = B.reshape(-1, B.shape[-1])  # bhd x n
        a_col_norms = torch.norm(A_, dim=0)  # n
        b_row_norms = torch.norm(B_, dim=0)  # n
        norm_mult = a_col_norms * b_row_norms # norm_mult: # n
        # A_top_k_dim Shape: k
        # B_top_k_dim Shape: k
        top_k_indices = topk(norm_mult, k, largest=True).indices
        top_k_indices, _ = torch.sort(top_k_indices) # top_k_indices Shape: k
        # A_top_k_dim = A[..., top_k_indices]
        B_top_k_dim = B_[..., top_k_indices].view(b, h, d, k).transpose(2, 3)

        return B_top_k_dim, top_k_indices

    else:
        a_col_norms = A.reshape(-1, n)  # bh x n
        if weight is None:
            b_row_norms = torch.norm(B, dim=-1).view(-1, n)  # bh x n
        else:
            b_row_norms = torch.norm(B * (weight.norm(dim=2)).unsqueeze(2), dim=-1).view(-1, n)  # bh x n

        if a_col_norms.sum() == 0.:
            norm_mult = b_row_norms
        else:
            norm_mult = a_col_norms * b_row_norms

        #### debiased sampling
        # # top_k_indices Shape: bh x k
        # # B_top_k_dim Shape: b x h x k x d
        sample_prob = (norm_mult + 1e-8) / (norm_mult.sum(dim=1).unsqueeze(dim=1) + 1e-8)
        top_k_indices = torch.multinomial(sample_prob, k, replacement=True)
        top_k_indices, _ = torch.sort(top_k_indices, dim=1)
        top_k_indices = top_k_indices.view(b, h, -1)
        B_top_k = torch.gather(B, dim=2, index=top_k_indices.unsqueeze(-1).expand(b, h, k, d))

        # residual_prob_sampled = torch.gather(residual_prob.view(b, h, n), dim=2, index=top_k_indices)
        # B_top_k_stoc_unbiased = (B_top_k_stoc + 1e-8) / (residual_prob_sampled.unsqueeze(dim=3) + 1e-8) / stoc_k * (1 - top_k_prob).view(b, h, 1, 1)

    return B_top_k, top_k_indices

    # stoc_weight = (1 - top_k_prob).view(b, h, 1, 1)
    # print(B_top_k_deter.shape, top_k_indices_deter.shape)
    # return B_top_k_deter, top_k_indices_deter, B_top_k_stoc_unbiased, top_k_indices, stoc_weight


class approxlinear_only_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias, ratio, deter_ratio, minimal_k, scheme, batch_dim_use_same_indices=True):

        scale = scheme.get_scale() if scheme is not None else None
        if ratio == 1.0 or scale is None:
            ctx.saved = input, None, weight, bias
        else:
            subsampled_input, top_k_indices = subsample_3D_input_by_norm(input, scale, ratio, deter_ratio, minimal_k, batch_dim_use_same_indices)
            ctx.saved = subsampled_input, top_k_indices, weight, bias

        ctx.scheme = scheme
        res = F.linear(input, weight, bias)
        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        subsampled_input, top_k_indices, weight, bias = ctx.saved

        grad_input = torch.matmul(grad_output, weight)
        if top_k_indices is None:
            grad_weight = torch.einsum('blo, bli->oi', grad_output, subsampled_input)
        else:
            # weight Shape: #d_output x #d_input
            # subsampled_input: n x L x k
            # grad_output: n x L x #d_output
            # top_k_indices Shape: k
            # grad_weight_: #d_output x k
            grad_output = grad_output.reshape(grad_output.shape[0] * grad_output.shape[1], -1)
            grad_output_ = grad_output[top_k_indices]
            grad_weight = torch.einsum('ko, ki->oi', grad_output_, subsampled_input)

        if bias is not None:
            grad_bias = grad_output.view(-1, grad_output.shape[-1]).sum(0)
        else:
            grad_bias = None

        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)

        del subsampled_input, top_k_indices, weight, bias

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


@torch.no_grad()
def subsample_3D_input_by_norm(input, grad_norm, sample_ratio, deter_ratio, minimal_k, batch_dim_use_same_indices: bool =True):
    """
    The shape of input tensor is BLD, where B is the batch size, L is the sentence length and D is the embedding dim.
    batch_dim_use_same_indices == False means each input sample could have custom top-k indices, thus the indices shape is
    B x k where k is the size after down-sampling (k << D). In contrast, batch_dim_use_same_indices == True
    means we enforce the top-k indices are the same along the B and L dims. And in this case, the shape of top-k indices is [k].
    """
    # A: input: n x L x #d_input
    # B: weight: #d_input x #d_output
    in_features = input.shape[0] * input.shape[1]
    k_candidate = int(in_features * sample_ratio)
    k = min(max(k_candidate, minimal_k), in_features)
    # print(k, deter_k, stoc_k)
    ## ===================================
    ## select only according to weight norm
    # norm_mult = b_row_norms
    ## ===================================

    if k == in_features:
        return input, None

    a_col_norms = torch.norm(input.view(-1, input.shape[-1]), dim=1)

    if grad_norm is None or grad_norm.sum() == 0.:
        # print(a_col_norms.shape, b_row_norms.shape, scale.shape)
        norm_mult = a_col_norms
    else:
        norm_mult = a_col_norms * grad_norm.view(-1)

    sample_prob = (norm_mult + 1e-8) / (norm_mult.sum() + 1e-8)
    top_k_indices = torch.multinomial(sample_prob, k, replacement=True)
    top_k_indices, _ = torch.sort(top_k_indices)
    input_top_k = input.view(in_features, -1)[top_k_indices, ...]

    # prob_sampled = sample_prob[top_k_indices].unsqueeze(dim=1)
    # input_top_k_stoc_unbiased = (input_top_k + 1e-8) / (prob_sampled + 1e-8) / stoc_k

    return input_top_k, top_k_indices



@torch.no_grad()
def weighted_subsample_4D_AB(A, B, weight, sample_ratio, minimal_k, batch_dim_use_same_indices: bool =True):
    # A: b x h x LA x d
    # B: b x h x d x LB
    # print(f'A: {A.shape}')
    # print(f'B: {B.shape}')
    b, h, lA, d = A.shape
    b, h, d, lB = B.shape
    k_candidate = int(d * sample_ratio)
    k = min(max(k_candidate, minimal_k), d)
    if k == d:
        return A, B, None

    if batch_dim_use_same_indices:
        raise NotImplementedError

    A_ = A.reshape(-1, lA, d) # bh x lA x d
    if weight.sum() == 0.:
        a_col_norms = torch.norm(A_, dim=1) # bh x d
    else:
        a_col_norms = torch.norm(A_*weight.view(b*h, -1).unsqueeze(-1), dim=1)
    B_ = B.reshape(-1, d, lB) # bh x lB x d
    b_row_norms = torch.norm(B_, dim=2) # bh x d
    norm_mult = a_col_norms * b_row_norms
    # norm_mult: # d if batch_dim_use_same_indices else bh x d

    # top_k_indices Shape: bh x k
    # A_top_k_dim Shape: b x h x lA x k
    # B_top_k_dim Shape: b x h x k x lB
    top_k_indices = topk(norm_mult, k, dim=1, largest=True).indices
    top_k_indices, _ = torch.sort(top_k_indices)
    A_top_k_dim = torch.gather(A_, dim=2, index=top_k_indices.unsqueeze(1).expand(b*h, lA, k)).reshape(b, h, lA, k)
    B_top_k_dim = B_[torch.arange(b*h).unsqueeze(1), top_k_indices].reshape(b, h, k, lB)

    return A_top_k_dim, B_top_k_dim, top_k_indices.view(b, h, -1)



@torch.no_grad()
def subsample_3D_input(input, weight, sample_ratio, minimal_k, scale=None, batch_dim_use_same_indices: bool =True):
    """
    The shape of input tensor is BLD, where B is the batch size, L is the sentence length and D is the embedding dim.
    batch_dim_use_same_indices == False means each input sample could have custom top-k indices, thus the indices shape is
    B x k where k is the size after down-sampling (k << D). In contrast, batch_dim_use_same_indices == True
    means we enforce the top-k indices are the same along the B and L dims. And in this case, the shape of top-k indices is [k].
    """
    # A: input: n x L x #d_input
    # B: weight: #d_input x #d_output

    in_features = weight.shape[0]
    k_candidate = int(in_features * sample_ratio)
    k = min(max(k_candidate, minimal_k), in_features)
    b_row_norms = torch.norm(weight, dim=1) # Shape #d_input


    ## ===================================
    ## select only according to weight norm
    # norm_mult = b_row_norms
    ## ===================================

    if batch_dim_use_same_indices:
        # Shape #d_input
        a_col_norms = torch.norm(input.view(-1, input.shape[-1]), dim=0)
    else:
        # Shape n x #d_input
        a_col_norms = torch.norm(input, dim=1)

    # norm_mult: # d_input if batch_dim_use_same_indices else n x # d_input

    norm_mult = a_col_norms * b_row_norms

    if batch_dim_use_same_indices:
        # top_k_indices Shape: k
        # input_top_k_dim Shape: n x L x k
        top_k_indices = topk(norm_mult, k, largest=True).indices
        top_k_indices, _ = torch.sort(top_k_indices)
        input_top_k_dim = input[..., top_k_indices]
    else:
        # top_k_indices Shape: n x k
        # gather_indices: n x L x k
        # input_top_k_dim: n x L x k
        top_k_indices = topk(norm_mult, k, dim=1, largest=True).indices
        # Very important
        top_k_indices, _ = torch.sort(top_k_indices)
        top_k_indices_for_gather = top_k_indices.unsqueeze(1).expand(top_k_indices.shape[0],
                                                                    input.shape[1],
                                                                    top_k_indices.shape[1])
        input_top_k_dim = torch.gather(input, dim=2, index=top_k_indices_for_gather)

    return input_top_k_dim, top_k_indices


def approx_linear_func_forward(input, weight, bias, sample_ratio, minimal_k):
    input_shape = input.shape
    C_in = input_shape[-1]
    input, weight = input.view(-1, C_in), weight

    # calculate the number of channels to sample for the forward propagation phase
    k_candidate = int(float(C_in) * sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate, minimal_k), C_in)

    # calculate norms of the columns of A and rows of B
    with torch.no_grad():
        # input: n x #d_input
        # weight: #d_input x #d_output
        in_features = weight.shape[0]
        k_candidate = int(in_features * sample_ratio)
        k = min(max(k_candidate, minimal_k), in_features)
        a_col_norms = torch.norm(input, dim=0)
        b_row_norms = torch.norm(weight, dim=0)
        norm_mult = a_col_norms * b_row_norms
        top_k_indices = topk(norm_mult, k, largest=True).indices
        # top_k_indices = torch.randperm(in_features, device='cuda')[:k]
        top_k_indices, _ = torch.sort(top_k_indices)
    A_top_k_cols = input[..., top_k_indices]
    B_top_k_rows = weight[..., top_k_indices]
    A_top_k_cols = A_top_k_cols.view(input_shape[0], input_shape[1], -1)
    return F.linear(A_top_k_cols, B_top_k_rows, bias=bias)
