import numpy as np

def compute_scale_pos_weight(y):
    """計算 scale_pos_weight"""
    neg, pos = np.bincount(y)
    print(f"[INFO] 正樣本數={pos}, 負樣本數={neg}")
    return (neg / pos) if pos > 0 else 1.0