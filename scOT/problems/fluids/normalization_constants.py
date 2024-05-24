import torch

CONSTANTS = {
    "mean": torch.tensor([0.80, 0.0, 0.0, 0.0]).unsqueeze(1).unsqueeze(1),
    "std": torch.tensor([0.31, 0.391, 0.356, 0.185]).unsqueeze(1).unsqueeze(1),
    "time": 20.0,
    "tracer_mean": 0.19586183,
    "tracer_std": 0.37,
}
