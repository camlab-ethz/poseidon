import torch
import h5py
import copy
from scOT.problems.base import BaseTimeDataset, BaseDataset
from scOT.problems.fluids.normalization_constants import CONSTANTS


class Airfoil(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 10869
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128

        data_path = self.data_path + "/SE-AF.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": 0.92984116,
            "std": 0.10864315,
        }

        self.input_dim = 1
        self.label_description = "[rho]"

        self.post_init()

    def __getitem__(self, idx):
        i = idx
        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, 0])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        labels = (
            torch.from_numpy(self.reader["solution"][i + self.start, 1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        labels = (labels - self.constants["mean"]) / self.constants["std"]

        pixel_mask = inputs == 1
        labels[pixel_mask] = 1

        return {
            "pixel_values": inputs,
            "labels": labels,
            "pixel_mask": pixel_mask,
        }


class RichtmyerMeshkov(BaseTimeDataset):
    def __init__(self, *args, tracer=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 1260
        self.N_val = 100
        self.N_test = 130
        self.resolution = 128

        data_path = self.data_path + "/CE-RM.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": torch.tensor([1.1964245, -7.164812e-06, 2.8968952e-06, 1.5648036])
            .unsqueeze(1)
            .unsqueeze(1),
            "std": torch.tensor([0.5543239, 0.24304213, 0.2430597, 0.89639103])
            .unsqueeze(1)
            .unsqueeze(1),
            "time": 20.0,
        }

        self.input_dim = 4
        self.label_description = "[rho],[u,v],[p]"

        self.pixel_mask = torch.tensor([False, False, False, False])

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )

        label = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class RayleighTaylor(BaseTimeDataset):
    def __init__(self, *args, tracer=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 10

        self.N_max = 1260
        self.N_val = 100
        self.N_test = 130
        self.resolution = 128

        data_path = self.data_path + "/GCE-RT.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": torch.tensor(
                [0.8970493, 4.0316996e-13, -1.3858967e-13, 0.7133829, -1.7055787]
            )
            .unsqueeze(1)
            .unsqueeze(1),
            "std": torch.tensor(
                [0.12857835, 0.014896976, 0.014896975, 0.21293919, 0.40131348]
            )
            .unsqueeze(1)
            .unsqueeze(1),
            "time": 10.0,
        }

        self.input_dim = 5
        self.label_description = "[rho],[u,v],[p],[g]"

        self.pixel_mask = torch.tensor([False, False, False, False, False])

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )

        g_1 = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1, 5:6])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )
        g_2 = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2, 5:6])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs = (inputs - self.constants["mean"][:4]) / self.constants["std"][:4]
        g_1 = (g_1 - self.constants["mean"][4]) / self.constants["std"][4]
        g_2 = (g_2 - self.constants["mean"][4]) / self.constants["std"][4]
        label = (label - self.constants["mean"][:4]) / self.constants["std"][:4]

        inputs = torch.cat([inputs, g_1], dim=0)
        label = torch.cat([label, g_2], dim=0)

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class CompressibleBase(BaseTimeDataset):
    def __init__(self, file_path, *args, tracer=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = 10000
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128
        self.tracer = tracer

        data_path = self.data_path + file_path
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = copy.deepcopy(CONSTANTS)

        self.input_dim = 4 if not tracer else 5
        self.label_description = (
            "[rho],[u,v],[p]" if not tracer else "[rho],[u,v],[p],[tracer]"
        )

        self.pixel_mask = (
            torch.tensor([False, False, False, False])
            if not tracer
            else torch.tensor([False, False, False, False, False])
        )

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader["data"][i + self.start, t1, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["data"][i + self.start, t2, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )

        inputs[3] = inputs[3] - self.mean_pressure
        label[3] = label[3] - self.mean_pressure

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        if self.tracer:
            input_tracer = (
                torch.from_numpy(self.reader["data"][i + self.start, t1, 4:5])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            output_tracer = (
                torch.from_numpy(self.reader["data"][i + self.start, t2, 4:5])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            inputs = torch.cat([inputs, input_tracer], dim=0)
            label = torch.cat([label, output_tracer], dim=0)

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class Gaussians(CompressibleBase):
    def __init__(self, *args, tracer=False, **kwargs):
        self.mean_pressure = 2.513
        file_path = "/CE-Gauss.nc"
        if tracer:
            raise NotImplementedError("Tracer not implemented for Gaussians")
        super().__init__(file_path, *args, tracer=tracer, **kwargs)


class KelvinHelmholtz(CompressibleBase):
    def __init__(self, *args, tracer=False, **kwargs):
        self.mean_pressure = 1.0
        file_path = "/CE-KH.nc"
        if tracer:
            raise NotImplementedError("Tracer not implemented for KelvinHelmholtz")
        super().__init__(file_path, *args, tracer=tracer, **kwargs)


class Riemann(CompressibleBase):
    def __init__(self, *args, tracer=False, **kwargs):
        self.mean_pressure = 0.215
        file_path = "/CE-RP.nc"
        if tracer:
            raise NotImplementedError("Tracer not implemented for Riemann")
        super().__init__(file_path, *args, tracer=tracer, **kwargs)


class RiemannCurved(CompressibleBase):
    def __init__(self, *args, tracer=False, **kwargs):
        self.mean_pressure = 0.553
        file_path = "/CE-CRP.nc"
        if tracer:
            raise NotImplementedError("Tracer not implemented for RiemannCurved")
        super().__init__(file_path, *args, tracer=tracer, **kwargs)


class RiemannKelvinHelmholtz(CompressibleBase):
    def __init__(self, *args, tracer=False, **kwargs):
        self.mean_pressure = 1.33
        file_path = "/CE-RPUI.nc"
        if tracer:
            raise NotImplementedError(
                "Tracer not implemented for RiemannKelvinHelmholtz"
            )
        super().__init__(file_path, *args, tracer=tracer, **kwargs)
