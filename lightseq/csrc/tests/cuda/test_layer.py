import sys
from __init__ import lightseq_dir

sys.path.insert(0, lightseq_dir)

import random
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
