import torch
import h5py
from scOT.problems.base import BaseTimeDataset


class AllenCahn(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 19

        self.N_max = 15000
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/ACE.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.002484262,
            "std": 0.65351176,
            "time": 19.0,
        }

        self.input_dim = 1
        self.label_description = "[u]"

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
        }
