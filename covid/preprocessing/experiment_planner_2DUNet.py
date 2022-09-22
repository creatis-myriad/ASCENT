import errno
import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np

from covid.models.components.unet import UNet
from covid.preprocessing.utils import get_pool_and_conv_props
from covid.utils.file_and_folder_operations import load_pickle


class nnUNetPlanner2D:
    """Plan experiment for 2D nnUNet.

    This planner is blatantly copied and slightly modified from nnUNet's ExperimentPlanner2D_v21.

    Ref:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/experiment_planning/experiment_planner_baseline_2DUNet_v21.py
    """

    def __init__(self, preprocessed_folder: Union[str, Path]) -> None:
        """
        Args:
            preprocessed_folder: Path to 'preprocessed' folder of a dataset.

        Raises:
            FileNotFoundError: If dataset_properties.pkl is not found in preprocessed_folder.
        """

        self.preprocessed_folder = preprocessed_folder
        if not os.path.isfile(os.path.join(self.preprocessed_folder, "dataset_properties.pkl")):
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                os.path.join(self.preprocessed_folder, "dataset_properties.pkl"),
            )

        self.dataset_properties = load_pickle(
            os.path.join(self.preprocessed_folder, "dataset_properties.pkl")
        )
        self.unet_base_num_features = 32
        self.unet_max_num_filters = 512
        self.unet_max_numpool = 999
        self.unet_featuremap_min_edge_length = 4
        self.conv_per_stage = 2
        self.unet_min_batch_size = 2
        self.how_much_of_a_patient_must_the_network_see_at_stage0 = 4  # 1/4 of a patient
        # all samples in the batch together cannot cover more than 5% of the entire dataset
        self.batch_size_covers_max_percent_of_dataset = 0.05

    def get_properties(
        self,
        median_shape: np.array,
        current_spacing: list,
        num_cases: int,
        num_classes: int,
        num_modalities: int,
    ) -> dict:
        """Compute training and model parameters based on nnUNet's heuristic rules.

        Args:
            median_shape: Median shape of dataset.
            current_spacing: Target spacing to resample data.
            num_cases: Number of cases in the dataset.
            num_classes: Number of label classes in the dataset.
            num_modalities: Number of modalities in the dataset.

        Returns:
            Plan dictionary containing:
                - Batch size
                - Number of pooling of axis
                - Input patch size
                - Median data shape in voxels
                - Pooling strides
                - Convolution kernels size
                - Dummy 2D augmentation flag.
        """

        dataset_num_voxels = np.prod(median_shape, dtype=np.int64) * num_cases
        input_patch_size = median_shape[:-1]

        (
            network_num_pool_per_axis,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            new_shp,
            shape_must_be_divisible_by,
        ) = get_pool_and_conv_props(
            current_spacing[:-1],
            input_patch_size,
            self.unet_featuremap_min_edge_length,
            self.unet_max_numpool,
        )

        # we pretend to use 30 feature maps. The larger memory footpring of 32 vs 30 is more than
        # offset by the fp16 training. We make fp16 training default.
        # Reason for 32 vs 30 feature maps is that 32 is faster in fp16 training (because multiple
        # of 8)
        ref = (
            UNet.use_this_for_batch_size_computation_2D * UNet.DEFAULT_BATCH_SIZE_2D / 2
        )  # for batch size 2
        here = UNet.compute_approx_vram_consumption(
            new_shp,
            network_num_pool_per_axis,
            30,
            self.unet_max_num_filters,
            num_modalities,
            num_classes,
            pool_op_kernel_sizes,
            conv_per_stage=self.conv_per_stage,
        )
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / median_shape[1:])[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = get_pool_and_conv_props(
                current_spacing[1:],
                tmp,
                self.unet_featuremap_min_edge_length,
                self.unet_max_numpool,
            )
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            (
                network_num_pool_per_axis,
                pool_op_kernel_sizes,
                conv_kernel_sizes,
                new_shp,
                shape_must_be_divisible_by,
            ) = get_pool_and_conv_props(
                current_spacing[1:],
                new_shp,
                self.unet_featuremap_min_edge_length,
                self.unet_max_numpool,
            )

            here = UNet.compute_approx_vram_consumption(
                new_shp,
                network_num_pool_per_axis,
                self.unet_base_num_features,
                self.unet_max_num_filters,
                num_modalities,
                num_classes,
                pool_op_kernel_sizes,
                conv_per_stage=self.conv_per_stage,
            )
            # print(new_shp)

        batch_size = int(np.floor(ref / here) * 2)
        input_patch_size = new_shp

        if batch_size < self.unet_min_batch_size:
            raise RuntimeError("This should not happen")

        # check if batch size is too large (more than 5 % of dataset)
        max_batch_size = np.round(
            self.batch_size_covers_max_percent_of_dataset
            * dataset_num_voxels
            / np.prod(input_patch_size, dtype=np.int64)
        ).astype(int)
        batch_size = max(1, min(batch_size, max_batch_size))

        plan = {
            "batch_size": batch_size,
            "num_pool_per_axis": network_num_pool_per_axis,
            "patch_size": input_patch_size,
            "median_patient_size_in_voxels": median_shape,
            "current_spacing": current_spacing,
            "pool_op_kernel_sizes": pool_op_kernel_sizes,
            "conv_kernel_sizes": conv_kernel_sizes,
            "do_dummy_2D_data_aug": False,
        }
        return plan

    def plan_experiment(self):
        """Plan experiment."""

        all_shapes_after_resampling = self.dataset_properties["all_shapes_after_resampling"]
        current_spacing = self.dataset_properties["spacing_after_resampling"]
        all_cases = self.dataset_properties["all_cases"]
        all_classes = self.dataset_properties["all_classes"]
        modalities = self.dataset_properties["modalities"]
        num_modalities = len(list(modalities.keys()))

        median_shape = np.median(np.vstack(all_shapes_after_resampling), 0).astype(int)
        print("The median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(all_shapes_after_resampling), 0)
        print("The max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(all_shapes_after_resampling), 0)
        print("The min shape in the dataset is ", min_shape)

        print(
            "We don't want feature maps smaller than ",
            self.unet_featuremap_min_edge_length,
            " in the bottleneck",
        )

        plan = self.get_properties(
            median_shape, current_spacing, len(all_cases), len(all_classes) + 1, num_modalities
        )

        print(plan)


if __name__ == "__main__":
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    preprocessed_folder = root / "data" / "CAMUS" / "preprocessed"
    planner = nnUNetPlanner2D(preprocessed_folder)
    planner.plan_experiment()
