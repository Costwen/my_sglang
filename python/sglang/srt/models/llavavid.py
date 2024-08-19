"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Inference-only LLaVa video model compatible with HuggingFace weights."""

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import math
from torch import nn
from transformers import LlavaConfig, AutoModel, AutoConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel, SiglipVisionConfig

# List all attributes in the module
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.mm_utils import (
    get_anyres_image_grid_shape,
    unpad_image,
    unpad_image_shape,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from .siglip_encoder import SigLipVisionTower


class LlavaVidForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = None

        if getattr(self.config, "vision_config", None) is None:
            self.config.vision_config = SiglipVisionConfig(self.config.mm_vision_tower)

        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = Qwen2Config(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        if getattr(self.config, "projector_hidden_act", None) is None:
            self.config.projector_hidden_act = "gelu"
        
        self.config.image_token_index = -200
    
        self.multi_modal_projector = LlavaMultiModalProjector(config).half()
        self.mm_spatial_pool_stride = getattr(self.config, "mm_spatial_pool_stride", 2)
        self.resampler = nn.AvgPool2d(
            kernel_size=self.mm_spatial_pool_stride, stride=self.mm_spatial_pool_stride
        )
        self.language_model = Qwen2ForCausalLM(config, quant_config=quant_config)
        self.num_frames = getattr(self.config, "num_frames", 16)
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )

    def pad_input_ids(self, input_ids, pad_value, pt_shape=None, image_size=None):
        
        resize_h, resize_w = self.num_patches_per_side // self.mm_spatial_pool_stride, self.num_patches_per_side // self.mm_spatial_pool_stride
        num_frames = pt_shape[0]
        new_image_feature_len = num_frames * (resize_h * (resize_w + 1))

        pad_ids = pad_value * new_image_feature_len 

        offset = input_ids.index(self.config.image_token_index)
        # old_len + pad_len - 1, because we need to remove image_token_id
        new_input_ids = (
            input_ids[:offset]
            + pad_ids[:new_image_feature_len]
            + input_ids[offset + 1 :]
        )
        return new_input_ids, offset

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values = torch.load("images.pt", map_location='cpu').to(pixel_values.device)

        # NOTE: This is not mebuild_vision_towermory efficient. (output_hidden_states=True) will save all the hidden stated.
        chunk_size = 2
        selected_image_feature = []
        device = pixel_values.device
        num_of_frames = pixel_values.shape[0]
    
        for i in range(0, num_of_frames, chunk_size):
            image_output = self.vision_tower(pixel_values[i:i + chunk_size])
            selected_image_feature.append(image_output)
        
        image_output = torch.cat(selected_image_feature, dim=0)

        # image_output = self.vision_tower(pixel_values)
        image_output = image_output.view(num_of_frames, self.num_patches_per_side, self.num_patches_per_side, -1)

        image_output = self.multi_modal_projector(image_output)

        image_output = image_output.permute(0, 3, 1, 2).contiguous()
        image_output = (
            self.resampler(image_output).permute(0, 2, 3, 1).contiguous()
        )
        selected_image_feature = image_output
    
        resize_h = selected_image_feature.shape[-2]

        image_feature = selected_image_feature.view(num_of_frames, 1, resize_h, resize_h, -1)
        
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)

        image_feature = torch.cat((image_feature, self.language_model.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        
        image_feature = image_feature.permute(2, 1, 0).contiguous()
        
        return image_feature

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        pixel_values: Optional[List[Optional[np.array]]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_offsets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if input_metadata.forward_mode == ForwardMode.EXTEND:
            bs = input_metadata.batch_size
            # Embed text input
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            # Embed vision input
            need_vision = (
                (positions[input_metadata.extend_start_loc] < self.image_feature_len)
                .cpu()
                .numpy()
            )
            # FIXME: We need to substract the length of the system prompt
            has_pixel = np.array([pixel_values[i] is not None for i in range(bs)])
            need_vision = need_vision & has_pixel

            if need_vision.any():
                pixel_values = [pixel_values[i] for i in range(bs) if need_vision[i]]
                image_sizes = [image_sizes[i] for i in range(bs) if need_vision[i]]

                ########## Encode Image ########

                image_features = []
                for pixel_value in pixel_values:
                    pixel_value = torch.tensor(pixel_value, device=self.vision_tower.device)                    
                    # tmp_feature = self.encode_images(pixel_value).flatten(0, 1)
                    # print(abs(image_feature - tmp_feature).mean())
                    # image_features.append(image_feature)
                    image_features.append(self.encode_images(pixel_value).flatten(0, 1))
                    
                
                extend_start_loc_cpu = input_metadata.extend_start_loc.cpu().numpy()
                pt = 0
                for i in range(bs):
                    if not need_vision[i]:
                        continue

                    start_idx = extend_start_loc_cpu[i]
                    pad_len, pad_dim = image_features[pt].shape  # 576, 4096
                    dim = input_embeds.shape[1]
                    assert (
                        pad_dim == dim
                    ), "invalid pad_dim={}, input_embed_dim={}!".format(pad_dim, dim)
                    # Fill in the placeholder for the image
                    try:
                        input_embeds[
                            start_idx
                            + image_offsets[i] : start_idx
                            + image_offsets[i]
                            + pad_len
                        ] = image_features[pt]
                    except RuntimeError as e:
                        print(f"RuntimeError in llava image encoding: {e}")
                        print(input_embeds.shape)
                        print(start_idx, image_offsets[i])
                    pt += 1

            return self.language_model(
                input_ids, positions, input_metadata, input_embeds=input_embeds
            )
        elif input_metadata.forward_mode == ForwardMode.DECODE:
            return self.language_model(input_ids, positions, input_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = self.config.mm_vision_tower

        # self.vision_tower = SiglipVisionModel.from_pretrained(vision_path, torch_dtype=torch.float16).cuda()
        # del self.vision_tower.vision_model.encoder.layers[-1:]
        # self.vision_tower.vision_model.head = nn.Identity()
        
        self.vision_tower = SigLipVisionTower(vision_path, vision_tower_cfg=self.config.vision_config).cuda().half()
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer

        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size


        self.image_feature_len = self.num_frames * int(
            (self.image_size / self.patch_size / self.mm_spatial_pool_stride) ** 2
        )
        if self.vision_feature_select_strategy == "patch":
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)

        # load mm_projector
        projector_weights = {
            "model.mm_projector.0": "multi_modal_projector.linear_1",
            "model.mm_projector.2": "multi_modal_projector.linear_2",
            "model.vision_resampler.mm_projector.0": "multi_modal_projector.linear_1",
            "model.vision_resampler.mm_projector.2": "multi_modal_projector.linear_2",
            "model.vision_tower.vision_tower": "vision_tower.vision_tower",  # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
        }
        params_dict = dict(self.named_parameters())
        weights = list(weights)

        for name, loaded_weight in weights:
            # FIXME: why projector weights read two times?
            if "projector" in name or "vision_tower" in name:
            # if "projector" in name in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                else:
                    print(f"Warning: {name} not found in the model")
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        self.language_model.load_weights(weights)

        # monkey_path_clip_vision_embed_forward()

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


first_call = True


def clip_vision_embed_forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    batch_size = pixel_values.shape[0]

    # Move this conv layer to CPU to avoid a bug in torch >= 2.1 on A10G.
    global first_call
    if first_call:
        self.patch_embedding.cpu().float()
        first_call = False
    pixel_values = pixel_values.to(dtype=torch.float32, device="cpu")
    patch_embeds = self.patch_embedding(pixel_values).cuda().half()

    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings


def monkey_path_clip_vision_embed_forward():
    import transformers

    setattr(
        transformers.models.siglip.modeling_siglip.SiglipVisionEmbeddings,
        "forward",
        clip_vision_embed_forward,
    )


EntryClass = LlavaVidForCausalLM
