import torch
import os
import h5py
import numpy as np
from scOT.problems.base import BaseDataset


class Helmholtz(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 19675
        self.N_val = 128
        self.N_test = 512
        self.resolution = 128

        self.file_path = os.path.join(
            self.data_path,
            "Helmholtz.h5",
        )
        self.file_path = self._move_to_local_scratch(self.file_path)
        self.reader = h5py.File(self.file_path, "r")
        self.mean = 0.11523915668552
        self.std = 0.8279975746000605

        self.input_dim = 2
        self.label_description = "[u]"

        self.post_init()

    def __getitem__(self, idx):
        inputs = (
            torch.from_numpy(self.reader["Sample_" + str(idx + self.start)]["a"][:])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        inputs = inputs - 1
        b = float(np.array(self.reader["Sample_" + str(idx + self.start)]["bc"]))
        bc = b * torch.ones_like(inputs)
        inputs = torch.cat((inputs, bc), dim=0)

        labels = (
            torch.from_numpy(self.reader["Sample_" + str(idx + self.start)]["u"][:])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (labels - self.mean) / self.std

        return {"pixel_values": inputs, "labels": labels}
