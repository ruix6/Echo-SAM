# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Echo_Sam,
    TwoWayTransformer,
)

def build_echo_sam_vit_b(checkpoint=None):
    return _build_echo_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

build_echo_sam = build_echo_sam_vit_b

echo_sam_model_registry = {
    "default": build_echo_sam_vit_b,
    "vit_b": build_echo_sam_vit_b,
}


def _build_echo_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    echo_sam = Echo_Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    echo_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            new_dict = load_from(echo_sam, state_dict, image_size, vit_patch_size)
            echo_sam.load_state_dict(new_dict)
        except:
            raise RuntimeError("Checkpoint is not compatible with the model.")

    return echo_sam

def load_from(us_sam, sam_dict, image_size, patch_size):
    """
    sam_dict with module.
    """
    new_dict = us_sam.state_dict()
    rel_pos_keys = [k[7:] for k, v in sam_dict.items() if 'rel_pos' in k and ('2' in k or '5' in k or '8' in k or '11' in k)]
    trained_weights_keys = [k[7:] for k, v in sam_dict.items() if k[7:] not in rel_pos_keys]
    token_size = int(image_size//patch_size)

    for name, _ in new_dict.items():
        if name in rel_pos_keys:
            temp_param = sam_dict['module.'+name]
            #print(name, 'modified loaded, size:', temp_param.shape)
            h, w = temp_param.shape
            temp_param = temp_param.unsqueeze(0).unsqueeze(0)
            temp_param = torch.nn.functional.interpolate(temp_param, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_dict[name] = temp_param.squeeze(0).squeeze(0)
        if name in trained_weights_keys:
            #print(name, 'loaded')
            new_dict[name] = sam_dict['module.'+name]


    print('Echo-SAM loaded')
    return new_dict

if __name__ == '__main__':
    model = echo_sam_model_registry['vit_b'](checkpoint=None)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    img = torch.randn(1, 3, 256, 256).to(device)
    box = torch.randn(1, 1, 4).to(device)
    mask = torch.randn(1, 1, 128, 128).to(device)
    points_prompts = torch.randn(1, 1, 2).to(device)
    points_labels = torch.as_tensor([[1]], dtype=torch.float32).to(device)

    img_encoder_output = model.image_encoder(img)
    print(img_encoder_output.shape)

    # output = model(img, mask, box, points_prompts, points_labels)
    output = model(img, None, box, points_prompts, points_labels)
    print(output.shape)
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)