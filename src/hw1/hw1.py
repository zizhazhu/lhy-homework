import numpy as np
import torch


# set a random seed for reproducibility
def set_no_rand(seed=42069):
    # 避免一些性能优化造成的不确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
