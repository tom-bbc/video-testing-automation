import yaml
import math
import torch
import decord
import numpy as np
from time import time
from tqdm import tqdm
from itertools import cycle
from decord import VideoReader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ExplainableVQA import open_clip
from ExplainableVQA.DOVER.dover import DOVER
from ExplainableVQA.DOVER.dover.datasets import spatial_temporal_view_decomposition, UnifiedFrameSampler
from ExplainableVQA.model import TextEncoder, MaxVQA, EnhancedVisualEncoder
