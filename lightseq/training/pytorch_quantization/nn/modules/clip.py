#
# Copyright (c) 2022 Bytedance Inc. All rights reserved.
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Implement a clip module as pytorch only has a simple clamp function """
import torch
from torch import nn
from torch.nn.parameter import Parameter

from lightseq.training.pytorch_quantization.nn import functional as QF

__all__ = ["Clip"]


class Clip(nn.Module):
    """Clip tensor

    Args:
        clip_value_min: A number or tensor of lower bound to clip
        clip_value_max: A number of tensor of upper bound to clip
        learn_min: A boolean. If True, learn min. clip_value_min will be used to initialize. Default False
        learn_max: A boolean. Similar as learn_min but for max.

    Raises:
        ValueError:
    """

    def __init__(
        self,
        clip_value_max,
        learn_max=False,
        is_embed=False,
        hz=None,
    ):
        super(Clip, self).__init__()
        self.is_embed = is_embed
        self.clip_value_max = Parameter(
            torch.tensor(clip_value_max),
            requires_grad=True
        )

    def forward(self, inputs):
        clip_value_max = self.clip_value_max
        clip_value_min = -clip_value_max
        outputs = torch.clamp(inputs, clip_value_min, clip_value_max)
        return outputs
