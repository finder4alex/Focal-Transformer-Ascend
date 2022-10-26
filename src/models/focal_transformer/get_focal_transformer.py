# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Get SwinTransformer of different size for args"""
import mindspore as ms
if ms.__version__ >= "1.6.0":
    print("************************************", flush=True)
    print("********** load high model *********", flush=True)
    print("************************************", flush=True)
    from .focal_transformer_v2_high import FocalTransformer
else:
    from .focal_transformer_v2 import FocalTransformer


def get_focaltransformer(args):
    """get swintransformer v2 according to args"""
    # override args
    image_size = args.image_size
    patch_size = args.patch_size
    in_chans = args.in_channel
    embed_dim = args.embed_dim
    depths = args.depths
    num_heads = args.num_heads
    window_size = args.window_size
    mlp_ratio = args.mlp_ratio
    qkv_bias = True
    drop_path_rate = args.drop_path_rate
    ape = args.ape
    patch_norm = args.patch_norm
    focal_stages = args.focal_stages
    focal_levels = args.focal_levels
    focal_windows = args.focal_windows
    expand_sizes = args.expand_sizes
    focal_topK = args.focal_topK
    focal_pool = args.focal_pool
    use_conv_embed = args.use_conv_embed
    print(25 * "=" + "MODEL CONFIG" + 25 * "=")
    print(f"==> IMAGE_SIZE:         {image_size}")
    print(f"==> PATCH_SIZE:         {patch_size}")
    print(f"==> IN_CHANS:           {in_chans}")
    print(f"==> NUM_CLASSES:        {args.num_classes}")
    print(f"==> EMBED_DIM:          {embed_dim}")
    print(f"==> DEPTHS:             {depths}")
    print(f"==> NUM_HEADS:          {num_heads}")
    print(f"==> WINDOW_SIZE:        {window_size}")
    print(f"==> MLP_RATIO:          {mlp_ratio}")
    print(f"==> QKV_BIAS:           {qkv_bias}")
    print(f"==> DROP_PATH_RATE:     {drop_path_rate}")
    print(f"==> APE:                {ape}")
    print(f"==> PATCH_NORM:         {patch_norm}")
    print(f"==> FOCAL_STAGES:       {focal_stages}")
    print(f"==> FOCAL_LEVELS:       {focal_levels}")
    print(f"==> FOCAL_WINDOWS:      {focal_windows}")
    print(f"==> EXPAND_SIZES:       {expand_sizes}")
    print(f"==> FOCAL_TOPK:         {focal_topK}")
    print(f"==> FOCAL_POOL:         {focal_pool}")
    print(f"==> USE_CONV_EMBED:         {use_conv_embed}")
    print(25 * "=" + "FINISHED" + 25 * "=")

    model = FocalTransformer(
        img_size=image_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=args.num_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        focal_stages=focal_stages,
        focal_levels=focal_levels,
        focal_windows=focal_windows,
        focal_topK=focal_topK,
        focal_pool=focal_pool,
        expand_sizes=expand_sizes,
        use_conv_embed=use_conv_embed)
    # print(model)

    return model


def focalv2_tiny_useconv_patch4_window7_224(args):
    """focalv2_tiny_useconv_patch4_window7_224"""
    return get_focaltransformer(args)


def focalv2_small_useconv_patch4_window7_224(args):
    """focalv2_small_useconv_patch4_window7_224"""
    return get_focaltransformer(args)


def focalv2_base_useconv_patch4_window7_224(args):
    """focalv2_base_useconv_patch4_window7_224"""
    return get_focaltransformer(args)
