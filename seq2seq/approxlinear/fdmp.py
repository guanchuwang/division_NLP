import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import autocast
# import cpp_extension.quantization as ext_quantization
# import exact.cpp_extension.quantization as ext_quantization
import seq2seq.backend.fdmp_cpp_extension.quantization as ext_quantization
# import cpp_extension.backward_func as ext_backward_func
# from torch.cuda.amp import autocast as autocast
import math

import time


total_act_mem = 0
total_act_mem_lfc = 0 # torch.tensor(0).type(torch.long)
total_act_mem_hfc = 0 # torch.tensor(0).type(torch.long)

# @torch.no_grad()
# def abs_window_size(N, window_size):
#     if config.round_window:
#         return round(window_size*N + 0.5)
#     else:
#         return round(window_size*N)

class FDMP(Function):

    @staticmethod
    @torch.no_grad()
    def quantize_and_pack1d(data, bits, mn, mx, N):

        mn_ = mn.view(N, 1, 1).repeat(1, data.shape[1], 1)
        mx_ = mx.view(N, 1, 1).repeat(1, data.shape[1], 1)

        # print(data.shape)
        # print(mn_.shape)
        # print(mx_.shape)
        # # print(scale.shape)
        # print(bits, type(bits))

        # output = pack_func(data, mn, mx, scale.to(data.dtype), bits, True)
        output, scale = ext_quantization.pack_single_precision(data, mn_, mx_, bits, True)
        scale = scale[:, 0, 0].clone()
        # import pdb
        # pdb.set_trace()

        return output, scale

    @staticmethod
    @torch.no_grad()
    def dequantize_and_unpack1d(data, shape, bits, scale, mn, max_thread=1024):

        # Pad to group_size
        Batch, Channel, feature_dim = shape

        # print(data.shape)
        # print(shape)
        # print(scale.shape)
        # print(mn.shape)

        if feature_dim > max_thread:
            thread_loop = math.ceil(feature_dim/max_thread)
            thread = feature_dim // thread_loop
            mn_ = mn.view(Batch,1,1).repeat(1, Channel*thread_loop, 1)
            scale_ = scale.view(Batch,1,1).repeat(1, Channel*thread_loop, 1)
            data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch, Channel*thread_loop, thread)

        else:
            mn_ = mn.view(Batch,1,1).repeat(1, Channel, 1)
            scale_ = scale.view(Batch,1,1).repeat(1, Channel, 1)
            data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch, Channel, feature_dim)

        return data

    @staticmethod
    @torch.no_grad()
    def fdmp1d(x, config):

        Batch, Channel, feature_dim = x.shape

        # print(Batch, Channel, feature_dim)

        if feature_dim == 1:
            return x, None, None, None, None

        pool_kernel_size = config.lfc_block if Channel >= config.lfc_block else Channel
        x_reshape = x.permute((0, 2, 1)) # Batch, feature_dim, Channel,
        x_lfc = F.avg_pool1d(x_reshape, pool_kernel_size, stride=pool_kernel_size, padding=0)
        x_lfc_float16 = x_lfc.to(torch.bfloat16)
        x_lfc_large = F.interpolate(x_lfc_float16.to(x_lfc.dtype), size=[Channel],
                                         scale_factor=None) # Batch, feature_dim, Channel/(block^2)

        x_hfc = (x_reshape - x_lfc_large).permute((0, 2, 1)) # Batch, Channel, feature_dim
        if feature_dim > config.max_thread:
            thread_loop = math.ceil(feature_dim/config.max_thread) # .type(torch.int)
            thread = feature_dim // thread_loop
            x_hfc_groups = x_hfc.reshape(Batch, Channel*thread_loop, thread)
        else:
            x_hfc_groups = x_hfc.clone() #

        q_min = x_hfc_groups.min(dim=-1).values.min(dim=-1).values
        mx = x_hfc_groups.max(dim=-1).values.max(dim=-1).values
        q_bits = config.hfc_bit_num

        # x_hfc_groups = x_hfc_groups.contiguous() if not x_hfc_groups.is_contiguous() else x_hfc_groups
        q_input, q_scale = FDMP.quantize_and_pack1d(x_hfc_groups, q_bits, q_min, mx, Batch)

        if_float32 = (x.dtype is torch.float32)

        return x_lfc_float16, q_input, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16), if_float32

    @staticmethod
    @torch.no_grad()
    def de_fdmp1d(feature_pack, q_input_shape, config):

        Batch, Channel, feature_dim = q_input_shape

        if feature_dim == 1:
            x, _, _, _, _ = feature_pack
            return x

        x_lfc_float16, q_input, q_scale, q_min, if_float32 = feature_pack

        # Estimate valid group size
        if if_float32 or (not config.half_precision):
            x_lfc = x_lfc_float16.to(torch.float32)  # Remove it if accuracy drops
            q_scale = q_scale.to(torch.float32)
            q_min = q_min.to(torch.float32)
        else:
            x_lfc = x_lfc_float16.to(torch.float16)  # Remove it if accuracy drops
            q_scale = q_scale.to(torch.float16)
            q_min = q_min.to(torch.float16) # bfloat16

        q_bits = config.hfc_bit_num

        x_hfc_dequant = FDMP.dequantize_and_unpack1d(q_input, q_input_shape, q_bits, q_scale, q_min)
        x_hfc_dequant = x_hfc_dequant.view(*q_input_shape)
        x_lfc_large = F.interpolate(x_lfc, size=[Channel], scale_factor=None).permute((0, 2, 1)) # Batch, Channel, feature_dim
        # print(x_lfc_large.shape)
        # print(x_hfc_dequant.shape)

        return x_lfc_large + x_hfc_dequant

    @staticmethod
    @torch.no_grad()
    def quantize_and_pack(data, bits, mn, mx, N):

        # Pack to bitstream
        # print(pack_func)
        # print(bits)
        # scale = (2 ** bits - 1) / (mx - mn)

        mn_ = mn.view(N, 1, 1).repeat(1, data.shape[1], 1)
        mx_ = mx.view(N, 1, 1).repeat(1, data.shape[1], 1)

        # print(data.shape)
        # print(mn_.shape)
        # print(mx_.shape)
        # # print(scale.shape)
        # print(bits, type(bits))

        # output = pack_func(data, mn, mx, scale.to(data.dtype), bits, True)
        output, scale = ext_quantization.pack_single_precision(data, mn_, mx_, bits, True)
        scale = scale[:,0,0].clone()
        # import pdb
        # pdb.set_trace()

        return output, scale

    @staticmethod
    @torch.no_grad()
    def dequantize_and_unpack(data, shape, bits, scale, mn, max_thread=1024):

        # Pad to group_size
        Batch, Channel, Higth, Width = shape
        num_features = int(shape[2:].numel())

        if num_features > max_thread:
            mn_ = mn.view(Batch * Channel, 1, 1).repeat(1, Higth, 1)
            scale_ = scale.view(Batch * Channel, 1, 1).repeat(1, Higth, 1) # N, num_features // group_size, group_size)
            data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch * Channel, Higth, Width)

        else:
            mn_ = mn.view(Batch * Channel, 1, 1)
            scale_ = scale.view(Batch * Channel, 1, 1)
            data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch * Channel, 1, Higth * Width)

        return data

    @staticmethod
    @torch.no_grad()
    def fdmp(x, config):

        Batch, Channel, Higth, Width = x.shape

        if Higth == 1:
            return x, None, None, None, None


        # print(Batch, Channel, Higth, Width)

        pool_ks_higth = config.lfc_block if Higth >= config.lfc_block else Higth
        pool_ks_width = config.lfc_block if Width >= config.lfc_block else Width
        x_lfc = F.avg_pool2d(x, kernel_size=(pool_ks_higth, pool_ks_width), stride=(pool_ks_higth, pool_ks_width), padding=0)
        x_lfc_float16 = x_lfc.to(torch.bfloat16)
        x_lfc_large = F.upsample_nearest(x_lfc_float16.to(x_lfc.dtype), size=(Higth, Width), scale_factor=None) # x_lfc.dtype

        # x_lfc_3d = x_lfc.reshape(Batch*Channel, x_lfc.shape[2], x_lfc.shape[3])
        # x_lfc_large_3da = F.interpolate(x_lfc_3d, size=(Width), mode='linear')
        # x_lfc_large_3db = F.interpolate(x_lfc_large_3da.permute(0,2,1), size=(Higth), mode='linear').permute(0,2,1)
        # x_lfc_large = x_lfc_large_3db.reshape(Batch, Channel, Higth, Width)
        # print(x.shape, x_lfc.shape, x_lfc_large.shape)

        x_hfc = x - x_lfc_large

        featuremap_area = Higth * Width # x_hfc.shape[-2:].numel()  # should be n

        if featuremap_area > config.max_thread:
            x_hfc_groups = x_hfc.reshape(Batch * Channel, Higth, Width)
        else:
            x_hfc_groups = x_hfc.reshape(Batch * Channel, 1, Higth * Width)

        q_min = x_hfc_groups.min(dim=-1).values.min(dim=-1).values
        mx = x_hfc_groups.max(dim=-1).values.max(dim=-1).values
        q_bits = config.hfc_bit_num
        q_input, q_scale = FDMP.quantize_and_pack(x_hfc_groups, q_bits, q_min, mx, Batch * Channel)

        if_float32 = (x.dtype is torch.float32)

        return x_lfc_float16, q_input, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16), if_float32

    @staticmethod
    @torch.no_grad()
    def de_fdmp(feature_pack, q_input_shape, config):

        # if window_size >= 1:
        #     x, _, _, _, _ = feature_pack
        #     return x

        Batch, Channel, Higth, Width = q_input_shape

        if Higth == 1:
            x, _, _, _, _ = feature_pack
            return x

        x_lfc_float16, q_input, q_scale, q_min, if_float32 = feature_pack

        # Estimate valid group size
        if if_float32 or (not config.half_precision):
            x_lfc = x_lfc_float16.to(torch.float32)  # Remove it if accuracy drops
            q_scale = q_scale.to(torch.float32)
            q_min = q_min.to(torch.float32)
        else:
            x_lfc = x_lfc_float16.to(torch.float16)  # Remove it if accuracy drops
            q_scale = q_scale.to(torch.float16)
            q_min = q_min.to(torch.float16) # bfloat16

        q_bits = config.hfc_bit_num

        x_hfc_dequant = FDMP.dequantize_and_unpack(q_input, q_input_shape, q_bits, q_scale, q_min)

        # Remove padding
        # num_features = q_input_shape[1:].numel()
        # x_hfc_dequant = x_hfc_dequant.view(q_input_shape[0], -1)[:, :num_features]
        x_hfc_dequant = x_hfc_dequant.view(*q_input_shape).contiguous()

        # pool_kernel_size = config.lfc_block if H >= config.lfc_block else H
        # x_lfc_large = F.interpolate(x_lfc, scale_factor=pool_kernel_size, mode='nearest')
        x_lfc_large = F.upsample_nearest(x_lfc, size=(Higth, Width), scale_factor=None)

        return x_lfc_large + x_hfc_dequant


# class FDMP(Function):
#
#     # @staticmethod
#     # @torch.no_grad()
#     # def no_scheme_compute_quantization_bits(input, group_size):
#     #     if not config.half_precision:
#     #         return FDMP_.no_scheme_compute_quantization_bits(input, group_size)
#     #     else:
#     #         with autocast():
#     #             return FDMP_.no_scheme_compute_quantization_bits(input, group_size)
#
#     @staticmethod
#     @torch.no_grad()
#     def quantize_and_pack(data, bits, mn, mx, N):
#         if not config.half_precision:
#             return FDMP_.quantize_and_pack(data, bits, mn, mx, N)
#         else:
#             with autocast():
#                 return FDMP_.quantize_and_pack(data, bits, mn, mx, N)
#
#     @staticmethod
#     @torch.no_grad()
#     def dequantize_and_unpack(data, shape, bits, scale, mn):
#         if not config.half_precision:
#             return FDMP_.dequantize_and_unpack(data, shape, bits, scale, mn)
#         else:
#             with autocast():
#                 return FDMP_.dequantize_and_unpack(data, shape, bits, scale, mn)
#
#     @staticmethod
#     @torch.no_grad()
#     def fdmp(x):
#         if not config.half_precision:
#             return FDMP_.fdmp(x)
#         else:
#             with autocast():
#                 return FDMP_.fdmp(x)
#
#     @staticmethod
#     @torch.no_grad()
#     def de_fdmp(feature_pack, q_input_shape):
#         if not config.half_precision:
#             return FDMP_.de_fdmp(feature_pack, q_input_shape)
#         else:
#             with autocast():
#                 return FDMP_.de_fdmp(feature_pack, q_input_shape)
#
#     @staticmethod
#     @torch.no_grad()
#     def quantize_and_pack1d(data, bits, mn, mx, N):
#         if not config.half_precision:
#             return FDMP_.quantize_and_pack1d(data, bits, mn, mx, N)
#         else:
#             with autocast():
#                 return FDMP_.quantize_and_pack1d(data, bits, mn, mx, N)
#
#     @staticmethod
#     @torch.no_grad()
#     def dequantize_and_unpack1d(data, shape, bits, scale, mn):
#         if not config.half_precision:
#             return FDMP_.dequantize_and_unpack1d(data, shape, bits, scale, mn)
#         else:
#             with autocast():
#                 return FDMP_.dequantize_and_unpack1d(data, shape, bits, scale, mn)
#
#     @staticmethod
#     @torch.no_grad()
#     def fdmp1d(x):
#         if not config.half_precision:
#             return FDMP_.fdmp1d(x)
#         else:
#             with autocast():
#                 return FDMP_.fdmp1d(x)
#
#     @staticmethod
#     @torch.no_grad()
#     def de_fdmp1d(feature_pack, q_input_shape):
#         if not config.half_precision:
#             return FDMP_.de_fdmp1d(feature_pack, q_input_shape)
#         else:
#             with autocast():
#                 return FDMP_.de_fdmp1d(feature_pack, q_input_shape)


