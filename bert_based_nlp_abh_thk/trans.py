import random

import numpy as np
import torch
from transformers import BertTokenizer

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
