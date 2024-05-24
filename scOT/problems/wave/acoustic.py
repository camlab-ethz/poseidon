import torch
import h5py
from scOT.problems.base import BaseTimeDataset


class Layer(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10512
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/Wave-Layer.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.03467443221585092,
            "std": 0.10442421752963911,
            "mean_c": 3498.5644380917424,
            "std_c": 647.843958567462,
            "time": 20.0,
        }

        self.input_dim = 2
        self.label_description = "[u],[c]"

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs_c = (
            torch.from_numpy(self.reader["c"][i + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        inputs_c = (inputs_c - self.constants["mean_c"]) / self.constants["std_c"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]

        inputs = torch.cat([inputs, inputs_c], dim=0)
        labels = torch.cat([labels, inputs_c], dim=0)

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
        }


class Gaussians(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 15

        self.N_max = 10512
        self.N_val = 60
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/Wave-Gauss.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.0334376316,
            "std": 0.1171879068,
            "mean_c": 2618.4593933,
            "std_c": 601.51658913,
            "time": 15.0,
        }

        self.input_dim = 2
        self.label_description = "[u],[c]"

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs_c = (
            torch.from_numpy(self.reader["c"][i + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        inputs_c = (inputs_c - self.constants["mean_c"]) / self.constants["std_c"]
        labels = (labels - self.constants["mean"]) / self.constants["std"]

        inputs = torch.cat([inputs, inputs_c], dim=0)
        labels = torch.cat([labels, inputs_c], dim=0)

        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": time,
        }
