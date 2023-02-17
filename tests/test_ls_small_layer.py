import random
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

from tests.util import TestDecorator
from lightseq.training.ops.pytorch import LayerBuilder

kt = TestDecorator()
layer_module = LayerBuilder().load()


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run()
