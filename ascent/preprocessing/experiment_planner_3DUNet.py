from copy import deepcopy
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from ascent import utils
from ascent.models.components.unet import UNet
from ascent.preprocessing.experiment_planner_2DUNet import nnUNetPlanner2D
from ascent.preprocessing.utils import get_pool_and_conv_props

log = utils.get_pylogger(__name__)


class nnUNetPlanner3D(nnUNetPlanner2D):
    """Plan experiment for 3D nnUNet.

    This planner is blatantly copied and slightly modified from nnUNet's ExperimentPlanner3D_v21.

    Ref:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/experiment_planning/experiment_planner_baseline_3DUNet_v21.py
    """

    def __init__(self, preprocessed_folder: Union[str, Path]) -> None:
        """Initialize class instance.

        Args:
            preprocessed_folder: Path to 'preprocessed' folder of a dataset.
        """
        super().__init__(preprocessed_folder)
        self.unet_max_num_filters = 320
        self.anisotropy_threshold = 3

    def get_properties(
        self,
        median_shape: np.array,
        current_spacing: list,
        num_cases: int,
        num_classes: int,
        num_modalities: int,
    ) -> dict[str, Union[int, bool, list[Sequence[int]], list[list[Sequence[int]]]]]:
        """Compute training and model parameters based on nnUNet's heuristic rules.

        Computation of 3D input patch shape is different from 2D. Instead of using directly the
        median image shape, an isotropic patch of 512x512x512 mm is created and then clipped to the
        median image shape.

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

        # compute how many voxels are one mm
        input_patch_size = 1 / np.array(current_spacing)

        # normalize voxels per mm
        input_patch_size /= input_patch_size.mean()

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
        input_patch_size = np.round(input_patch_size).astype(int)

        # clip it to the median shape of the dataset because patches larger then that make not much
        # sense
        input_patch_size = [min(i, j) for i, j in zip(input_patch_size, median_shape)]

        (
            network_num_pool_per_axis,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            new_shp,
            shape_must_be_divisible_by,
        ) = get_pool_and_conv_props(
            current_spacing,
            input_patch_size,
            self.unet_featuremap_min_edge_length,
            self.unet_max_numpool,
        )

        # we pretend to use 30 feature maps. The larger memory footpring of 32 vs 30 is more than
        # offset by the fp16 training. We make fp16 training default.
        # Reason for 32 vs 30 feature maps is that 32 is faster in fp16 training (because multiple
        # of 8)
        ref = (
            UNet.use_this_for_batch_size_computation_3D
            * self.unet_base_num_features
            / UNet.BASE_NUM_FEATURES_3D
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
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = get_pool_and_conv_props(
                current_spacing,
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
                current_spacing,
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

        input_patch_size = new_shp

        batch_size = UNet.DEFAULT_BATCH_SIZE_3D  # This is what works with 128**3
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large
        max_batch_size = np.round(
            self.batch_size_covers_max_percent_of_dataset
            * dataset_num_voxels
            / np.prod(input_patch_size, dtype=np.int64)
        ).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = int(max(1, min(batch_size, max_batch_size)))

        do_dummy_2D_data_aug = (
            bool(max(input_patch_size) / input_patch_size[0]) > self.anisotropy_threshold
        )

        pool_op_kernel_sizes = [[1] * len(input_patch_size)] + pool_op_kernel_sizes

        plan = {
            "batch_size": batch_size,
            "num_pool_per_axis": network_num_pool_per_axis,
            "patch_size": input_patch_size,
            "median_patient_size_in_voxels": median_shape,
            "current_spacing": current_spacing,
            "do_dummy_2D_data_aug": do_dummy_2D_data_aug,
            "pool_op_kernel_sizes": pool_op_kernel_sizes,
            "conv_kernel_sizes": conv_kernel_sizes,
        }

        return plan

    def plan_experiment(self) -> None:
        """Plan experiment and write plans to yaml files."""
        log.info("Planning experiment for 3D U-Net...")
        all_shapes_after_resampling = self.dataset_properties["all_shapes_after_resampling"]
        current_spacing = self.dataset_properties["spacing_after_resampling"]
        all_cases = self.dataset_properties["all_cases"]
        all_classes = self.dataset_properties["all_classes"]
        modalities = self.dataset_properties["modalities"]
        num_modalities = len(list(modalities.keys()))

        median_shape = np.median(np.vstack(all_shapes_after_resampling), 0).astype(int)
        log.info(f"The median shape of the dataset is {median_shape}.")

        max_shape = np.max(np.vstack(all_shapes_after_resampling), 0)
        log.info(f"The max shape in the dataset is {max_shape}.")
        min_shape = np.min(np.vstack(all_shapes_after_resampling), 0)
        log.info(f"The min shape in the dataset is {min_shape}.")

        log.info(
            f"We don't want feature maps smaller than {self.unet_featuremap_min_edge_length} in the"
            f" bottleneck.",
        )

        plan = self.get_properties(
            median_shape, current_spacing, len(all_cases), len(all_classes) + 1, num_modalities
        )

        if not plan["patch_size"].tolist()[-1] == 1:
            log.info(f"{plan}\n")
            self.write_plans_to_yaml(plan)
        else:
            log.info(
                "The 3D input patch size has a singleton depth dimension. 2D U-Net is more than "
                "enough. Not generating 3D plans...."
            )


if __name__ == "__main__":
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    preprocessed_folder = root / "data" / "CAMUS" / "preprocessed"
    planner = nnUNetPlanner3D(preprocessed_folder)
    planner.plan_experiment()
