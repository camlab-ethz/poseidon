"""
This file contains the dataset selector get_dataset, as well as the base 
classes for all datasets.
"""

from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, List, Dict
from abc import ABC
import re
import os
import shutil
from accelerate.utils import broadcast_object_list


def get_dataset(dataset, **kwargs):
    """
    Get a dataset by name.
    If you enter a list of str, will return a ConcatDataset of the datasets.

    Available choices are:
    - fluids.incompressible.BrownianBridge(.tracer)
    - fluids.incompressible.Gaussians(.tracer)
    - fluids.incompressible.ShearLayer
    - fluids.incompressible.Sines(.tracer)
    - fluids.incompressible.PiecewiseConstants(.tracer)
    - fluids.incompressible.VortexSheet(.tracer)
    - fluids.incompressible.forcing.KolmogorovFlow
    - fluids.compressible.gravity.RayleighTaylor(.tracer)
    - fluids.compressible.RiemannKelvinHelmholtz
    - fluids.compressible.RiemannCurved
    - fluids.compressible.Riemann
    - fluids.compressible.KelvinHelmholtz
    - fluids.compressible.Gaussians
    - fluids.compressible.RichtmyerMeshkov(.tracer)
    - fluids.compressible.steady.Airfoil(.time)
    - elliptic.poisson.Gaussians(.time)
    - elliptic.Helmholtz(.time)
    - wave.Layer
    - wave.Gaussians
    - reaction_diffusion.AllenCahn

    Adding .out at the end of the str, returns a dataset with more time steps.
    **kwargs overwrite the default settings.
    .time is a time-wrapped time-independent dataset.
    """
    if isinstance(dataset, list):
        return ConcatDataset([get_dataset(d, **kwargs) for d in dataset])
    if "fluids" in dataset:
        if "fluids.incompressible" in dataset:
            if "BrownianBridge" in dataset:
                from .fluids.incompressible import BrownianBridge as dset
            elif "Gaussians" in dataset:
                from .fluids.incompressible import Gaussians as dset
            elif "ShearLayer" in dataset:
                from .fluids.incompressible import ShearLayer as dset
            elif "Sines" in dataset:
                from .fluids.incompressible import Sines as dset
            elif "PiecewiseConstants" in dataset:
                from .fluids.incompressible import PiecewiseConstants as dset
            elif "VortexSheet" in dataset:
                from .fluids.incompressible import VortexSheet as dset
            elif "forcing" in dataset:
                if "KolmogorovFlow" in dataset:
                    from .fluids.incompressible import KolmogorovFlow as dset
                else:
                    raise ValueError(f"Unknown dataset {dataset}")
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        elif "fluids.compressible" in dataset:
            if "gravity" in dataset:
                if "RayleighTaylor" in dataset:
                    from .fluids.compressible import RayleighTaylor as dset

                    if "out" in dataset:
                        default_time_settings = {
                            "max_num_time_steps": 10,
                            "time_step_size": 1,
                        }
                    else:
                        default_time_settings = {
                            "max_num_time_steps": 7,
                            "time_step_size": 1,
                        }
                    kwargs = {**default_time_settings, **kwargs}
                elif "Blast" in dataset:
                    from .fluids.compressible import Blast as dset
            elif "RiemannKelvinHelmholtz" in dataset:
                from .fluids.compressible import RiemannKelvinHelmholtz as dset
            elif "RiemannCurved" in dataset:
                from .fluids.compressible import RiemannCurved as dset
            elif "Riemann" in dataset:
                from .fluids.compressible import Riemann as dset
            elif "KelvinHelmholtz" in dataset:
                from .fluids.compressible import KelvinHelmholtz as dset
            elif "Gaussians" in dataset:
                from .fluids.compressible import Gaussians as dset
            elif "RichtmyerMeshkov" in dataset:
                from .fluids.compressible import RichtmyerMeshkov as dset
            elif "steady" in dataset:
                if "steady.Airfoil" in dataset:
                    from .fluids.compressible import Airfoil as dset

                    if "out" in dataset:
                        raise ValueError(f"Unknown dataset {dataset}")
                else:
                    raise ValueError(f"Unknown dataset {dataset}")
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        else:
            raise ValueError(f"Unknown dataset {dataset}")
        if "out" in dataset:
            default_time_settings = {"max_num_time_steps": 10, "time_step_size": 2}
        else:
            default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
        if "tracer" in dataset:
            tracer = True
        else:
            tracer = False
        if not "steady" in dataset:
            kwargs = {"tracer": tracer, **default_time_settings, **kwargs}
    elif "elliptic" in dataset:
        if ".out" in dataset:
            raise NotImplementedError(f"Unknown dataset {dataset}")
        if "elliptic.poisson" in dataset:
            if "Gaussians" in dataset:
                from .elliptic.poisson import Gaussians as dset
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        elif "elliptic.Helmholtz" in dataset:
            from .elliptic.helmholtz import Helmholtz as dset
        else:
            raise ValueError(f"Unknown dataset {dataset}")
    elif "wave" in dataset:
        if "wave.Layer" in dataset:
            if "out" in dataset:
                default_time_settings = {"max_num_time_steps": 10, "time_step_size": 2}
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .wave.acoustic import Layer as dset
        elif "wave.Gaussians" in dataset:
            if "out" in dataset:
                raise ValueError(f"Unknown dataset {dataset}")
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .wave.acoustic import Gaussians as dset
        else:
            raise ValueError(f"Unknown dataset {dataset}")
    elif "reaction_diffusion" in dataset:
        if "reaction_diffusion.AllenCahn" in dataset:
            if "out" in dataset:
                default_time_settings = {"max_num_time_steps": 9, "time_step_size": 2}
            else:
                default_time_settings = {"max_num_time_steps": 7, "time_step_size": 2}
            kwargs = {**default_time_settings, **kwargs}
            from .reaction_diffusion.allen_cahn import AllenCahn as dset
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return dset(**kwargs) if ".time" not in dataset else TimeWrapper(dset(**kwargs))


class BaseDataset(Dataset, ABC):
    """A base class for all datasets. Can be directly derived from if you have a steady/non-time dependent problem."""

    def __init__(
        self,
        which: Optional[str] = None,
        num_trajectories: Optional[int] = None,
        data_path: Optional[str] = "./data",
        move_to_local_scratch: Optional[str] = None,
    ) -> None:
        """
        Args:
            which: Which dataset to use, i.e. train, val, or test.
            resolution: The resolution of the dataset.
            num_trajectories: The number of trajectories to use for training.
            data_path: The path to the data files.
            move_to_local_scratch: If not None, move the data to this directory at dataset initialization and use it from there.
        """
        assert which in ["train", "val", "test"]
        assert num_trajectories is not None and (
            num_trajectories > 0 or num_trajectories in [-1, -2, -8]
        )

        self.num_trajectories = num_trajectories
        self.data_path = data_path
        self.which = which
        self.move_to_local_scratch = move_to_local_scratch

    def _move_to_local_scratch(self, file_path):
        if self.move_to_local_scratch is not None:
            data_dir = os.path.join(self.data_path, file_path)
            file = file_path.split("/")[-1]
            scratch_dir = self.move_to_local_scratch
            dest_dir = os.path.join(scratch_dir, file)
            RANK = int(os.environ.get("LOCAL_RANK", -1))
            if not os.path.exists(dest_dir) and (RANK == 0 or RANK == -1):
                print(f"Start copying {file} to {dest_dir}...")
                shutil.copy(data_dir, dest_dir)
                print("Finished data copy.")
            # idk how to do the barrier differently
            ls = broadcast_object_list([dest_dir], from_process=0)
            dest_dir = ls[0]
            return dest_dir
        else:
            return file_path

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        if self.num_trajectories == -1:
            self.num_trajectories = self.N_max - self.N_val - self.N_test
        elif self.num_trajectories == -2:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 2
        elif self.num_trajectories == -8:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 8
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        if self.which == "train":
            self.length = self.num_trajectories
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test
            self.start = self.N_max - self.N_test

        self.output_dim = self.label_description.count(",") + 1
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list

    def __len__(self) -> int:
        """
        Returns: overall length of dataset.
        """
        return self.length

    def __getitem__(self, idx) -> Dict:
        """
        Get an item. OVERWRITE!

        Args:
            idx: The index of the sample to get.

        Returns:
            A dict of key-value pairs of data.
        """
        pass

    @staticmethod
    def get_channel_lists(label_description):
        matches = re.findall(r"\[([^\[\]]+)\]", label_description)
        channel_slice_list = [0]  # use as channel_slice_list[i]:channel_slice_list[i+1]
        beautiful_descriptors = []
        for match in matches:
            channel_slice_list.append(channel_slice_list[-1] + 1 + match.count(","))
            splt = match.split(",")
            if len(splt) > 1:
                beautiful_descriptors.append("".join(splt))
            else:
                beautiful_descriptors.append(match)
        return beautiful_descriptors, channel_slice_list


class BaseTimeDataset(BaseDataset, ABC):
    """A base class for time dependent problems. Inherit time-dependent problems from here."""

    def __init__(
        self,
        *args,
        max_num_time_steps: Optional[int] = None,
        time_step_size: Optional[int] = None,
        fix_input_to_time_step: Optional[int] = None,
        allowed_time_transitions: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            max_num_time_steps: The maximum number of time steps to use.
            time_step_size: The size of the time step.
            fix_input_to_time_step: If not None, fix the input to this time step.
            allowed_time_transitions: If not None, only allow these time transitions (time steps).
        """
        assert max_num_time_steps is not None and max_num_time_steps > 0
        assert time_step_size is not None and time_step_size > 0
        assert fix_input_to_time_step is None or fix_input_to_time_step >= 0

        super().__init__(*args, **kwargs)
        self.max_num_time_steps = max_num_time_steps
        self.time_step_size = time_step_size
        self.fix_input_to_time_step = fix_input_to_time_step
        self.allowed_time_transitions = allowed_time_transitions

    def _idx_map(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1) + self.fix_input_to_time_step
            t = t2 - t1
        return i, t, t1, t2

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        self.max_time_step must have already been set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        if self.num_trajectories == -1:
            self.num_trajectories = self.N_max - self.N_val - self.N_test
        elif self.num_trajectories == -2:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 2
        elif self.num_trajectories == -8:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 8
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        assert self.max_num_time_steps is not None and self.max_num_time_steps > 0

        if self.fix_input_to_time_step is not None:
            self.multiplier = self.max_num_time_steps
        else:
            self.time_indices = []
            for i in range(self.max_num_time_steps + 1):
                for j in range(i, self.max_num_time_steps + 1):
                    if (
                        self.allowed_time_transitions is not None
                        and (j - i) not in self.allowed_time_transitions
                    ):
                        continue
                    self.time_indices.append(
                        (self.time_step_size * i, self.time_step_size * j)
                    )
            self.multiplier = len(self.time_indices)

        if self.which == "train":
            self.length = self.num_trajectories * self.multiplier
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test

        self.output_dim = self.label_description.count(",") + 1
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list


class TimeWrapper(BaseTimeDataset):
    """For time-independent problems to be plugged into time-dependent models."""

    def __init__(self, dataset):
        super().__init__(
            dataset.which,
            dataset.num_trajectories,
            dataset.data_path,
            None,
            max_num_time_steps=1,
            time_step_size=1,
        )
        self.dataset = dataset
        self.resolution = dataset.resolution
        self.input_dim = dataset.input_dim
        self.output_dim = dataset.output_dim
        self.channel_slice_list = dataset.channel_slice_list
        self.printable_channel_description = dataset.printable_channel_description

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {**self.dataset[idx], "time": 1.0}
