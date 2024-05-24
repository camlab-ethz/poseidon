import torch
import os
import h5py
from scOT.problems.base import BaseDataset

CONSTANTS = {
    "mean_source": 0.014822142414492256,
    "std_source": 4.755138816607612,
    "mean_solution": 0.0005603458434937093,
    "std_solution": 0.02401226126952699,
}


class Gaussians(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_max = 20000
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128

        self.file_path = os.path.join(self.data_path, "Poisson-Gauss.nc")
        self.file_path = self._move_to_local_scratch(self.file_path)
        self.reader = h5py.File(self.file_path, "r")
        self.constants = CONSTANTS

        self.input_dim = 1
        self.label_description = "[u]"

        self.post_init()

    def __getitem__(self, idx):
        inputs = (
            torch.from_numpy(self.reader["source"][idx + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        labels = (
            torch.from_numpy(self.reader["solution"][idx + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean_source"]) / self.constants["std_source"]
        labels = (labels - self.constants["mean_solution"]) / self.constants[
            "std_solution"
        ]

        return {"pixel_values": inputs, "labels": labels}
