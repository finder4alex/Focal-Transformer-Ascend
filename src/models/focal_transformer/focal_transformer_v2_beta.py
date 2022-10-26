# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Focal Transformer v2 beta """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from mindspore import nn
from mindspore import numpy
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common import initializer as weight_init
from mindspore.common import Parameter, Tensor

# from src.args import args
from src.models.focal_transformer.misc import _ntuple, Identity, DropPath1D

to_2tuple = _ntuple(2)


def np_topk(a, k, axis=-1, largest=True, sorted=True):
    """infer to https://zhuanlan.zhihu.com/p/374269641"""
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def np_gather(a, dim, index):
    expanded_index = [
        index if dim == i else np.arange(a.shape[i]).reshape(
            [-1 if i == j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
    return a[tuple(expanded_index)]


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = np.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Roll(nn.Cell):
    """Roll Cell"""

    def __init__(self, shift_size, shift_axis=(1, 2)):
        super(Roll, self).__init__()
        self.shift_size = to_2tuple(shift_size)
        self.shift_axis = shift_axis

    def construct(self, x):
        x = numpy.roll(x, self.shift_size, self.shift_axis)
        return x


class WindowPartitionConstruct(nn.Cell):
    """WindowPartitionConstruct Cell"""

    def __init__(self):
        super(WindowPartitionConstruct, self).__init__()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = self.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
        x = self.transpose(x, (0, 1, 3, 2, 4, 5))
        x = self.reshape(x, (B * H * W // (window_size ** 2), window_size, window_size, C))

        return x


class WindowPartitionNoReshapeConstruct(nn.Cell):
    """WindowPartitionNoReshapeConstruct Cell"""

    def __init__(self):
        super(WindowPartitionNoReshapeConstruct, self).__init__()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size
        Returns:
            windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
        """
        B_, H_, W_, C_ = x.shape
        pad_l = pad_t = 0
        pad_r = (window_size - W_ % window_size) % window_size
        pad_b = (window_size - H_ % window_size) % window_size
        if pad_r > 0 or pad_b > 0:
            print("NoReshape pad", flush=True)
            pad_op = ops.Pad(((0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)))
            x = pad_op(x)

        # B, H, W, C = x.shape
        # print(B_, H_, W_, C_, flush=True)
        # print(pad_t, pad_b, pad_l, pad_r, flush=True)
        # print(B, H, W, C, flush=True)
        B = B_
        H = H_ + pad_t + pad_b
        W = W_ + pad_l + pad_r
        C = C_
        x = self.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
        x = self.transpose(x, (0, 1, 3, 2, 4, 5))

        # return x
        return x, H // window_size, W // window_size


class WindowReverseConstruct(nn.Cell):
    """WindowReverseConstruct Cell"""

    def __init__(self):
        super(WindowReverseConstruct, self).__init__()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        # B = windows.shape[0] // (H * W / window_size / window_size)  # GRAPH_MODE
        B = int(windows.shape[0] // (H * W / window_size / window_size))  # PYNATIVE_MODE
        x = self.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
        x = self.transpose(x, (0, 1, 3, 2, 4, 5))
        x = self.reshape(x, (B, H, W, -1))

        return x


def get_topk_closest_indice(q_windows, k_windows, topk=1):
    # get pair-wise relative position index for each token inside the window
    coords_h_q = np.arange(q_windows[0])
    coords_w_q = np.arange(q_windows[1])

    if q_windows[0] != k_windows[0]:
        factor = k_windows[0] // q_windows[0]
        coords_h_q = coords_h_q * factor + factor // 2
        coords_w_q = coords_w_q * factor + factor // 2
    else:
        factor = 1

    coords_h_q_size = q_windows[0]
    coords_w_q_size = q_windows[1]
    coords_h_q = coords_h_q.reshape(coords_h_q_size, 1).repeat(
        coords_w_q_size, 1).reshape(coords_h_q_size, coords_w_q_size)
    coords_w_q = coords_w_q.reshape(1, coords_w_q_size).repeat(
        coords_h_q_size, 1).reshape(coords_w_q_size, coords_h_q_size).transpose(1, 0)
    coords_q = np.stack((coords_h_q, coords_w_q), axis=0)  # 2, Wh_q, Ww_q

    coords_h_k_size = k_windows[0]
    coords_w_k_size = k_windows[1]
    coords_h_k = np.arange(k_windows[0])
    coords_w_k = np.arange(k_windows[1])
    coords_h_k = coords_h_k.reshape(coords_h_k_size, 1).repeat(
        coords_w_k_size, 1).reshape(coords_h_k_size, coords_w_k_size)
    coords_w_k = coords_w_k.reshape(1, coords_w_k_size).repeat(
        coords_h_k_size, 1).reshape(coords_w_k_size, coords_h_k_size).transpose(1, 0)
    coords_k = np.stack((coords_h_k, coords_w_k), axis=0)  # 2, Wh, Ww

    coords_flatten_q = coords_q.reshape(2, -1)  # 2, Wh_q*Ww_q
    coords_flatten_k = coords_k.reshape(2, -1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    # print(relative_coords.shape, flush=True)

    relative_position_dists = np.sqrt(
        relative_coords[0].astype(np.float32) ** 2 + relative_coords[1].astype(np.float32) ** 2)

    topk = min(topk, relative_position_dists.shape[1])
    topk_score_k, topk_index_k = np_topk(-relative_position_dists, topk, axis=1)  # B, topK, nHeads
    indice_topk = topk_index_k
    relative_coord_topk = np_gather(
        relative_coords, 2, np.tile(np.expand_dims(indice_topk, axis=0), (2, 1, 1)))

    # topk_op = ops.TopK(sorted=True)
    # topk_score_k, topk_index_k = topk_op(Tensor(-relative_position_dists, dtype=mstype.float32), topk)
    # indice_topk = topk_index_k.asnumpy()
    # gather_op = ops.GatherD()
    # relative_coord_topk = gather_op(
    #     Tensor(relative_coords), 2,
    #     Tensor(np.tile(np.expand_dims(indice_topk, axis=0), (2, 1, 1)))).asnumpy()

    return indice_topk, relative_coord_topk.transpose(1, 2, 0), topk


class WindowAttention(nn.Cell):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, expand_size, shift_size, window_size, window_size_glo,
                 focal_window, focal_level, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., pool_method="none", topK=64):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.window_size_glo = window_size_glo
        self.pool_method = pool_method
        self.input_resolution = input_resolution  # NWh, NWw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.nWh, self.nWw = self.input_resolution[0] // self.window_size[0], \
            self.input_resolution[1] // self.window_size[1]

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj = nn.Dense(dim, dim, has_bias=True)
        self.proj_drop = nn.Dropout(1 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

        self.topK = topK

        h_size = self.window_size[0]
        w_size = self.window_size[1]
        coords_h_window = np.arange(self.window_size[0]) - self.window_size[0] // 2
        coords_w_window = np.arange(self.window_size[1]) - self.window_size[1] // 2
        coords_h_window = coords_h_window.reshape(h_size, 1).repeat(
            w_size, 1).reshape(h_size, w_size)
        coords_w_window = coords_w_window.reshape(1, w_size).repeat(
            h_size, 1).reshape(w_size, h_size).transpose(1, 0)
        coords_window = np.stack((coords_h_window, coords_w_window), axis=-1)
        self.window_coords = Parameter(Tensor(coords_window, dtype=mstype.int32), requires_grad=False)

        self.coord2rpb_all = nn.CellList()

        self.topks = []
        self.topk_closest_indice_shapes = []
        self.topk_coords_k_shapes = []
        self.topk_cloest_indice_params = []
        self.topk_cloest_coords_params = []
        print("============", flush=True)
        for k in range(self.focal_level):
            if k == 0:
                range_h = self.input_resolution[0]
                range_w = self.input_resolution[1]
            else:
                range_h = self.nWh
                range_w = self.nWw

            # build relative position range
            topk_closest_indice, topk_closest_coord, topK_updated = get_topk_closest_indice(
                (self.nWh, self.nWw), (range_h, range_w), self.topK)
            self.topks.append(topK_updated)

            if k > 0:
                # scaling the coordinates for pooled windows
                topk_closest_coord = topk_closest_coord * self.window_size[0]
            topk_closest_coord_window = np.expand_dims(
                topk_closest_coord, axis=1) + coords_window.reshape(-1, 2)[None, :, None, :]

            # self.__setattr__(
            #     "topk_cloest_indice_{}".format(k),
            #     Parameter(Tensor(topk_closest_indice, dtype=mstype.int32), requires_grad=False))
            # self.__setattr__(
            #     "topk_cloest_coords_{}".format(k),
            #     Parameter(Tensor(topk_closest_coord_window, dtype=mstype.float32), requires_grad=False))
            self.topk_cloest_indice_params.append(
                Parameter(
                    Tensor(topk_closest_indice, dtype=mstype.int32), requires_grad=False))
            self.topk_cloest_coords_params.append(
                Parameter(
                    Tensor(topk_closest_coord_window, dtype=mstype.float32), requires_grad=False))

            self.topk_closest_indice_shapes.append(topk_closest_indice.shape)
            self.topk_coords_k_shapes.append(topk_closest_coord_window.shape)
            print("indice", topk_closest_indice.shape, flush=True)
            print("coord_k", topk_closest_coord_window.shape, flush=True)

            coord2rpb = nn.SequentialCell(
                nn.Dense(2, head_dim),
                nn.ReLU(),
                nn.Dense(head_dim, self.num_heads))
            self.coord2rpb_all.append(coord2rpb)

        s0, s1, s2, s3 = 0, 0, 0, 0
        for a, b, c, d in self.topk_coords_k_shapes:
            s0 += a
            s1 += b
            s2 += c
            s3 += d
        self.top_rgb_cat_shape = (s0 // focal_level, s1 // focal_level, s2, self.num_heads)

        # operations
        self.batch_matmul_qk = ops.BatchMatMul(transpose_b=True)
        self.batch_matmul_v = ops.BatchMatMul()
        self.cat = ops.Concat(axis=2)
        self.gather = ops.GatherD()
        self.reshape = ops.Reshape()
        self.split = ops.Split(axis=2, output_num=3)
        self.squeeze = ops.Squeeze(axis=2)
        self.transpose = ops.Transpose()
        self.window_partition = WindowPartitionConstruct()

    def construct(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x_all[0]

        B, nH, nW, C = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (B, nH, nW, 3, C))
        qkv = self.transpose(qkv, (3, 0, 1, 2, 4))
        q, k, v = ops.Split(axis=0, output_num=3)(qkv)
        q = ops.Squeeze(axis=0)(q)
        k = ops.Squeeze(axis=0)(k)
        v = ops.Squeeze(axis=0)(v)
        # q = qkv[0]
        # k = qkv[1]
        # v = qkv[2]

        # partition q map
        q_windows = self.window_partition(q, self.window_size[0])
        q_windows = self.reshape(
            q_windows, (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))
        q_windows = self.transpose(q_windows, (0, 2, 1, 3))

        k_all = []
        v_all = []
        topKs = []
        topk_rpbs = []
        attn_cat_shape = 0
        for l_k in range(self.focal_level):
            topk_closest_indice_shape = self.topk_closest_indice_shapes[l_k]
            # topk_closest_indice = self.__getattr__("topk_cloest_indice_{}".format(l_k))
            topk_closest_indice = self.topk_cloest_indice_params[l_k]
            topk_indice_k = self.reshape(topk_closest_indice, (1, -1))
            topk_indice_k = numpy.tile(topk_indice_k, (B, 1))

            # topk_coords_k = self.__getattr__("topk_cloest_coords_{}".format(l_k))
            topk_coords_k = self.topk_cloest_coords_params[l_k]

            topk_rpb_k = self.coord2rpb_all[l_k](topk_coords_k)
            topk_rpbs.append(topk_rpb_k)

            if l_k == 0:
                k_k = self.reshape(k, (B, -1, self.num_heads, C // self.num_heads))
                v_k = self.reshape(v, (B, -1, self.num_heads, C // self.num_heads))
            else:
                x_k = x_all[l_k]
                qkv_k = self.qkv(x_k)
                qkv_k = self.reshape(qkv_k, (B, -1, 3, self.num_heads, C // self.num_heads))
                q_k, k_k, v_k = self.split(qkv_k)
                k_k = self.squeeze(k_k)
                v_k = self.squeeze(v_k)
                # k_k = qkv_k[:, :, 1]
                # v_k = qkv_k[:, :, 2]

            topk_indice_k = self.reshape(topk_indice_k, (B, -1, 1, 1))
            topk_indice_k = numpy.tile(topk_indice_k, (1, 1, self.num_heads, C // self.num_heads))
            k_k_selected = self.gather(k_k, 1, topk_indice_k)
            v_k_selected = self.gather(v_k, 1, topk_indice_k)

            # [B, x, y, num_heads, C // num_heads]
            selected_shape_0 = (B,) + topk_closest_indice_shape + (self.num_heads, C // self.num_heads)
            selected_shape_1 = (-1, self.num_heads, topk_closest_indice_shape[1], C // self.num_heads)

            k_k_selected = self.reshape(k_k_selected, selected_shape_0)
            k_k_selected = self.transpose(k_k_selected, (0, 1, 3, 2, 4))
            v_k_selected = self.reshape(v_k_selected, selected_shape_0)
            v_k_selected = self.transpose(v_k_selected, (0, 1, 3, 2, 4))

            k_k_selected = self.reshape(k_k_selected, selected_shape_1)
            v_k_selected = self.reshape(v_k_selected, selected_shape_1)

            k_all.append(k_k_selected)
            v_all.append(v_k_selected)
            topKs.append(topk_closest_indice_shape[1])
            attn_cat_shape += topk_closest_indice_shape[1]

        k_all = self.cat(k_all)
        v_all = self.cat(v_all)

        # N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        attn = self.batch_matmul_qk(q_windows, k_all)
        # window_area = self.window_size[0] * self.window_size[1]
        # window_area_whole = k_all.shape[2]

        attn_shape = (-1, self.num_heads, self.window_size[0] * self.window_size[0], attn_cat_shape)

        top_rgb_cat = self.cat(topk_rpbs)
        top_rgb_cat = self.transpose(top_rgb_cat, (0, 3, 1, 2))
        # top_rgb_cat = self.reshape(top_rgb_cat, (1,) + top_rgb_cat.shape)
        # top_rgb_cat = self.reshape(top_rgb_cat, (1,) + self.top_rgb_cat_shape)
        # top_rgb_cat = numpy.tile(top_rgb_cat, (B, 1, 1, 1, 1))
        # why use as bellow, infer to https://gitee.com/mindspore/mindspore/issues/I4Z8QX
        top_rgb_cat = numpy.tile(top_rgb_cat, (B, 1, 1, 1))
        top_rgb_cat = self.reshape(top_rgb_cat, (B,) + self.top_rgb_cat_shape)
        # top_rgb_cat = self.reshape(top_rgb_cat, attn.shape)
        top_rgb_cat = self.reshape(top_rgb_cat, attn_shape)
        attn = attn + top_rgb_cat

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.batch_matmul_v(attn, v_all)
        x = self.transpose(x, (0, 2, 1, 3))
        # x = self.reshape(x, x.shape[:2] + (-1,))
        # x = self.reshape(x, x.shape[:2] + (self.dim,))
        x = self.reshape(x, (-1, self.window_size[0] * self.window_size[0], self.dim))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FocalTransformerBlock(nn.Cell):
    r""" Focal Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none",
                 focal_level=1, focal_window=1, topK=64, use_layerscale=False, layerscale_value=1e-4):
        super(FocalTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.expand_size = 0
            # self.focal_level = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_size_glo = self.window_size

        self.pool_layers = nn.CellList()
        if self.pool_method != "none":
            for k in range(self.focal_level-1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Dense(window_size_glo * window_size_glo, 1))
                    weight_shape = self.pool_layers[-1].weight.shape
                    weight_dtype = self.pool_layers[-1].weight.dtype
                    weight_data = np.ones(weight_shape) * 1 / (window_size_glo * window_size_glo)
                    self.pool_layers[-1].weight.set_data(Tensor(weight_data, dtype=weight_dtype))
                    bias_shape = self.pool_layers[-1].bias.shape
                    bias_dtype = self.pool_layers[-1].bias.dtype
                    bias_data = np.zeros(bias_shape)
                    self.pool_layers[-1].bias.set_data(Tensor(bias_data, dtype=bias_dtype))
                    # self.pool_layers[-1].weight.data.fill_(1./(window_size_glo * window_size_glo))
                    # self.pool_layers[-1].bias.data.fill_(0)
                elif self.pool_method == "conv":
                    self.pool_layers.append(
                        nn.Conv2d(dim, dim, kernel_size=window_size_glo, stride=window_size_glo,
                                  pad_mode='valid', has_bias=True, groups=dim))

        self.norm1 = norm_layer((dim,) if isinstance(dim, int) else dim, epsilon=1e-5)
        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, expand_size=self.expand_size, shift_size=self.shift_size,
            window_size=to_2tuple(self.window_size), window_size_glo=to_2tuple(self.window_size_glo),
            focal_window=focal_window, focal_level=self.focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            pool_method=pool_method, topK=topK)

        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,) if isinstance(dim, int) else dim, epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            attn_mask = np.expand_dims(attn_mask, axis=1)
            attn_mask = np.expand_dims(attn_mask, axis=0)
            attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False)
            self.roll_pos = Roll(self.shift_size)
            self.roll_neg = Roll(-self.shift_size)
        else:
            self.attn_mask = None

        if self.use_layerscale:
            self.gamma_1 = Parameter(Tensor(layerscale_value * np.ones(dim)), requires_grad=True)
            self.gamma_2 = Parameter(Tensor(layerscale_value * np.ones(dim)), requires_grad=True)

        self.window_size_glo_list = []
        self.pooled_h_list = []
        self.pooled_w_list = []
        H, W = self.input_resolution
        for k in range(self.focal_level - 1):
            window_size_glo = math.floor(self.window_size_glo / (2 ** k))
            pooled_h = math.ceil(H / self.window_size) * (2 ** k)
            pooled_w = math.ceil(W / self.window_size) * (2 ** k)
            self.window_size_glo_list.append(window_size_glo)
            self.pooled_h_list.append(pooled_h)
            self.pooled_w_list.append(pooled_w)

        # operations
        self.mean = ops.ReduceMean(keep_dims=False)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.window_partition_noreshape = WindowPartitionNoReshapeConstruct()
        self.window_reverse = WindowReverseConstruct()

    def construct(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = self.reshape(x, (B, H, W, C))

        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
        else:
            shifted_x = x

        x_windows_all = [shifted_x]
        x_windows_masks_all = [self.attn_mask]

        if self.focal_level > 1 and self.pool_method != "none":
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level - 1):
                window_size_glo = self.window_size_glo_list[k]
                pooled_h = self.pooled_h_list[k]
                pooled_w = self.pooled_w_list[k]
                H_pool = pooled_h * window_size_glo
                W_pool = pooled_w * window_size_glo

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    print("pad h no", flush=True)
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    print("pad h", pad_t, pad_b, flush=True)
                    pad_op_h = ops.Pad(((0, 0), (pad_t, pad_b), (0, 0), (0, 0)))
                    x_level_k = pad_op_h(x_level_k)
                else:
                    print("pad h nothing", flush=True)

                if W > W_pool:
                    print("pad w no", flush=True)
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    print("pad w", pad_l, pad_r, flush=True)
                    pad_op_w = ops.Pad(((0, 0), (0, 0), (pad_l, pad_r), (0, 0)))
                    x_level_k = pad_op_w(x_level_k)
                else:
                    print("pad w nothing", flush=True)

                # B, nw, nw, window_size, window_size, C
                # x_windows_noreshape = self.window_partition_noreshape(x_level_k, window_size_glo)
                # nWh, nWw = x_windows_noreshape.shape[1:3]
                x_windows_noreshape, nWh, nWw = self.window_partition_noreshape(x_level_k, window_size_glo)
                if self.pool_method == "mean":
                    x_windows_pooled = self.mean(x_windows_noreshape, (3, 4))  # B, nWh, nWw, C
                    x_windows_all += [x_windows_pooled]
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2).max(-2)
                    x_windows_pooled = self.reshape(x_windows_pooled, (B, nWh, nWw, C))  # B, nWh, nWw, C
                    x_windows_all += [x_windows_pooled]
                elif self.pool_method == "fc":
                    # B, nWh, nWw, C, wsize**2
                    x_windows_noreshape = self.reshape(
                        x_windows_noreshape, (B, nWh, nWw, window_size_glo * window_size_glo, C))
                    x_windows_noreshape = self.transpose(x_windows_noreshape, (0, 1, 2, 4, 3))
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape)
                    x_windows_pooled = self.reshape(x_windows_pooled, (B, nWh, nWw, -1))  # B, nWh, nWw, C
                    x_windows_all += [x_windows_pooled]
                elif self.pool_method == "conv":
                    x_windows_noreshape = self.reshape(x_windows_noreshape, (-1, window_size_glo, window_size_glo, C))
                    # B * nw * nw, C, wsize, wsize
                    x_windows_noreshape = self.transpose(x_windows_noreshape, (0, 3, 1, 2))
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape)
                    x_windows_pooled = self.reshape(x_windows_pooled, (B, nWh, nWw, C)) # B, nWh, nWw, C
                    x_windows_all += [x_windows_pooled]

                x_windows_masks_all += [None]

        print("************", flush=True)
        for tmp in x_windows_all:
            print(tmp.shape, flush=True)
        print("************", flush=True)
        attn_windows = self.attn(x_windows_all, mask_all=x_windows_masks_all)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = self.reshape(attn_windows, (-1, self.window_size, self.window_size, C))
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
        else:
            x = shifted_x
        x = self.reshape(x, (B, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(
            self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))

        return x


class PatchMering(nn.Cell):
    r""" Patch Merging Layer.
    Args:
        img_size (tuple[int]): Resolution of input feature.
        in_chans (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, img_size, patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=nn.LayerNorm,
                 use_pre_norm=False, is_stem=False):
        super(PatchMering, self).__init__()
        self.input_resolution = img_size
        self.dim = in_chans
        self.reduction = nn.Dense(4 * in_chans, 2 * in_chans, has_bias=False)
        self.norm = norm_layer((4 * in_chans,), epsilon=1e-5)

        # operations
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape

        x = self.transpose(x, (0, 2, 3, 1))

        x = self.reshape(x, (B, H // 2, 2, W // 2, 2, C))
        x = self.transpose(x, (0, 1, 3, 4, 2, 5))
        x = self.reshape(x, (B, -1, 4 * C))

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Cell):
    """ A basic Focal Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, expand_size, expand_layer,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none",
                 focal_level=1, focal_window=1, topK=64, use_conv_embed=False, use_shift=False, use_pre_norm=False,
                 downsample=None, use_checkpoint=False, use_layerscale=False, layerscale_value=1e-4):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1

        # build blocks
        self.blocks = nn.CellList([
            FocalTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                expand_size=0 if (i % 2 == expand_factor) else expand_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pool_method=pool_method,
                focal_level=focal_level,
                focal_window=focal_window,
                topK=topK,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, patch_size=2, in_chans=dim, embed_dim=2*dim,
                use_conv_embed=use_conv_embed, norm_layer=norm_layer, use_pre_norm=use_pre_norm,
                is_stem=False
            )
        else:
            self.downsample = None

        # operations
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            # x = self.reshape(x, (x.shape[0], self.input_resolution[0], self.input_resolution[1], -1))
            x = self.reshape(x, (-1, self.input_resolution[0], self.input_resolution[1], self.dim))
            x = self.transpose(x, (0, 3, 1, 2))
            x = self.downsample(x)

        return x


class PatchEmbed(nn.Cell):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False,
                 norm_layer=None, use_pre_norm=False, is_stem=False):
        super(PatchEmbed, self).__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm
        self.use_conv_embed = use_conv_embed

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.kernel_size = kernel_size
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=kernel_size, stride=stride,
                pad_mode="pad", padding=padding, has_bias=True)
        else:
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode='valid', has_bias=True)

        if self.use_pre_norm:
            if norm_layer is not None:
                self.pre_norm = nn.GroupNorm(1, in_chans)
            else:
                self.pre_norm = None

        if norm_layer is not None:
            self.norm = norm_layer((embed_dim,) if isinstance(embed_dim, int) else embed_dim, epsilon=1e-5)
        else:
            self.norm = None

        # operations
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.use_pre_norm:
            x = self.pre_norm(x)

        x = self.proj(x)
        x = self.reshape(x, (B, self.embed_dim, -1))
        x = self.transpose(x, (0, 2, 1))
        if self.norm is not None:
            x = self.norm(x)

        return x


class FocalTransformer(nn.Cell):
    r""" Focal Transformer: Focal Self-attention for Local-Global Interactions in Vision Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        use_shift (bool): Whether to use window shift proposed by Swin Transformer. We observe that using shift or not does not make difference to our Focal Transformer. Default: False
        focal_stages (list): Which stages to perform focal attention. Default: [0, 1, 2, 3], means all stages
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1]
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1]
        expand_stages (list): Which stages to expand the finest grain window. Default: [0, 1, 2, 3], means all stages
        expand_sizes (list): The expand size for the finest grain level. Default: [3, 3, 3, 3]
        expand_layer (str): Which layers we want to expand the window for the finest grain leve. This can save computational and memory cost without the loss of performance. Default: "all"
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance, but we do not use it by default. Default: False
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False
        layerscale_value (float): Value for layer scale. Default: 1e-4
        use_pre_norm (bool): Whether use pre-norm in patch merging/embedding layer to control the feature magtigute. Default: False
    """
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 use_shift=False,
                 focal_stages=[0, 1, 2, 3],
                 focal_levels=[1, 1, 1, 1],
                 focal_windows=[7, 5, 3, 1],
                 focal_topK=64,
                 focal_pool="fc",
                 expand_stages=[0, 1, 2, 3],
                 expand_sizes=[3, 3, 3, 3],
                 expand_layer="all",
                 use_conv_embed=False,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 use_pre_norm=False,
                 **kwargs):
        super(FocalTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size), patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            use_conv_embed=use_conv_embed, is_stem=True,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = Parameter(Tensor(np.zeros((1, num_patches, embed_dim)), dtype=mstype.float32))

        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               pool_method=focal_pool if i_layer in focal_stages else "none",
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer],
                               focal_window=focal_windows[i_layer],
                               topK=focal_topK,
                               expand_size=expand_sizes[i_layer],
                               expand_layer=expand_layer,
                               use_conv_embed=use_conv_embed,
                               use_shift=use_shift,
                               use_pre_norm=use_pre_norm,
                               use_checkpoint=use_checkpoint,
                               use_layerscale=use_layerscale,
                               layerscale_value=layerscale_value)
            self.layers.append(layer)

        self.norm = norm_layer(
            (self.num_features,) if isinstance(self.num_features, int) else self.num_features, epsilon=1e-5)
        # diu'nei'lou'mou, nn.AdaptiveAvgPool1d(1) equals to ReduceMean
        # infer to https://blog.csdn.net/hymn1993/article/details/122780617
        self.avgpool = ops.ReduceMean(keep_dims=True)

        self.head = nn.Dense(
            in_channels=self.num_features, out_channels=num_classes, has_bias=True) if num_classes > 0 else Identity()

        # operations
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

        self.init_weights()

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    weight_init.initializer(
                        weight_init.TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(
                    weight_init.initializer(weight_init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(
                    weight_init.initializer(weight_init.Zero(), cell.beta.shape, cell.beta.dtype))

    def construct_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.transpose(x, (0, 2, 1))  # B C L
        x = self.avgpool(x, 2)  # B C 1
        x = self.reshape(x, (B, -1))
        return x

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


def main():
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE)
    # context.set_context(mode=context.GRAPH_MODE)
    # context.set_context(enable_graph_kernel=False)

    img_size = 224
    x = Tensor(np.random.rand(4, 3, img_size, img_size), dtype=mstype.float32)

    # focal tiny
    model = FocalTransformer(
        img_size=img_size, patch_size=4, embed_dim=96, drop_path_rate=0.2,
        depths=[2,2,6,2], num_heads=[3, 6, 12, 24], window_size=7,use_shift=False,
        focal_stages=[0, 1, 2, 3], focal_levels=[2,2,2,2], focal_windows=[7,5,3,1],
        focal_topK=128, focal_pool="fc", expand_sizes=[3,3,3,3], expand_layer="all",
        use_conv_embed=True)

    # focal small
    # model = FocalTransformer(
    #     img_size=img_size, patch_size=4, drop_path_rate=0.3, embed_dim=96,
    #     depths=[2,2,18,2], num_heads=[3,6,12,24], window_size=7, use_shift=False,
    #     focal_stages=[0,1,2,3], focal_levels=[2,2,2,2], focal_windows=[7,5,3,1],
    #     focal_topK=128, focal_pool="fc", expand_sizes=[3,3,3,3], expand_layer="all",
    #     use_conv_embed=True)

    y = model(x)
    print(y.shape, flush=True)


if __name__ == "__main__":
    main()
