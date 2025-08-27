import numpy as np
import os
import random
import torch

def set_global_seed(seed):
    """
    Make calculations reproducible by setting RANDOM seeds
    :param seed:
    :return:
    """
    # set the global variable to the new var throughout
    global SEED
    SEED = seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)