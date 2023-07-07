import os
from copy import deepcopy
from pathlib import Path
from typing import Callable, Tuple, Union

import hydra
import numpy as np
from lightning import LightningModule, Trainer
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ToTensord
from omegaconf import DictConfig

from ascent import utils
from ascent.train import AscentTrainer
from ascent.utils.file_and_folder_operations import load_pickle, subfiles
from ascent.utils.transforms import Preprocessd

log = utils.get_pylogger(__name__)


class AscentPredictor(AscentTrainer):
    """Ascent predictor that runs the main predict loop using Lightning Trainer."""

    @staticmethod
    def check_input_folder_and_return_datalist(
        input_folder: Union[str, Path],
        output_folder: Union[str, Path],
        overwrite_existing: bool,
        expected_num_modalities: int,
    ) -> list[dict[str, str]]:
        """Analyze input folder and convert the nifti files to datalist.

        Example of output: [{"image":""././path"}].

        Args:
            input_folder: Folder containing nifti files for inference.
            output_folder: Output folder to save predictions.
            overwrite_existing: Whether to overwrite existing predictions in output folder.
            expected_num_modalities: Number of input modalities.

        Returns:
            Datalist. [{"image": "././path"},]

        Raises:
            ValueError: If the input folder is empty.
            RuntimeError: If there are missing files in the input folder.
        """

        log.info(f"This model expects {expected_num_modalities} input modalities for each image.")
        files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

        maybe_case_ids = np.unique([i[:-12] for i in files])

        remaining = deepcopy(files)
        missing = []

        if not len(files) > 0:
            raise ValueError(
                "Input folder did not contain any images (expected to find .nii.gz file endings)"
            )

        # now check if all required files are present and that no unexpected files are remaining
        for c in maybe_case_ids:
            for n in range(expected_num_modalities):
                expected_output_file = c + "_%04.0d.nii.gz" % n
                if not os.path.isfile(os.path.join(input_folder, expected_output_file)):
                    missing.append(expected_output_file)
                else:
                    remaining.remove(expected_output_file)

        log.info(
            f"Found {len(maybe_case_ids)} unique case ids, here are some examples: "
            f"{np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10))}."
        )

        log.info(
            "If they don't look right, make sure to double check your filenames. They must end with "
            "_0000.nii.gz etc."
        )

        if len(remaining) > 0:
            log.warning(
                f"Found {len(remaining)} unexpected remaining files in the folder. Here are some "
                f"examples: {np.random.choice(remaining, min(len(remaining), 10))}."
            )

        if len(missing) > 0:
            log.warning(f"Some files are missing: {missing}")
            raise RuntimeError("Missing files in input_folder")

        os.makedirs(output_folder, exist_ok=True)

        # check if potential output files already existed in the output folder and remove the existing
        # case ids from datalist
        if overwrite_existing:
            if os.listdir(output_folder):
                output_files = subfiles(output_folder, suffix=".nii.gz", join=False, sort=True)
                if output_files:
                    existing_case_ids = [case[:-7] for case in output_files]
                    for case in existing_case_ids:
                        if case in maybe_case_ids:
                            index = np.argwhere(maybe_case_ids == case)
                            maybe_case_ids = np.delete(maybe_case_ids, index)

        all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
        list_of_lists = [
            [
                os.path.join(input_folder, i)
                for i in all_files
                if i[: len(j)].startswith(j) and len(i) == (len(j) + 12)
            ]
            for j in maybe_case_ids
        ]

        datalist = [{"image": image_paths} for image_paths in list_of_lists]

        return datalist

    @staticmethod
    def get_predict_transforms(dataset_properties: dict) -> Callable:
        """Build transforms compose to read and preprocess inference data.

        Args:
            dataset_properties: Properties used for preprocessing, eg. intensity properties.

        Returns:
            Monai Compose(transforms)
        """
        load_transforms = [
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
        ]

        sample_transforms = [
            Preprocessd(
                keys="image",
                target_spacing=dataset_properties["spacing_after_resampling"],
                intensity_properties=dataset_properties["intensity_properties"],
                do_resample=dataset_properties["do_resample"],
                do_normalize=dataset_properties["do_normalize"],
                modalities=dataset_properties["modalities"],
            ),
            ToTensord(keys="image", track_meta=True),
        ]

        return Compose(load_transforms + sample_transforms)

    @staticmethod
    @hydra.main(version_base="1.3", config_path="configs", config_name="predict")
    @utils.task_wrapper
    def run_system(cfg: DictConfig) -> Tuple[dict, dict]:
        """Predict unseen cases with a given checkpoint.

        Currently, this method only supports inference for nnUNet models.

        This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
        failure. Useful for multiruns, saving info about the crash, etc.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.

        Returns:
            Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.

        Raises:
            ValueError: If the checkpoint path is not provided.
        """
        # apply extra utilities
        # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
        utils.extras(cfg)

        if not cfg.ckpt_path:
            raise ValueError("ckpt_path must not be empty!")

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

        object_dict = {
            "cfg": cfg,
            "model": model,
            "trainer": trainer,
        }

        dataset_properties = load_pickle(
            os.path.join(
                cfg.paths.data_dir,
                cfg.dataset,
                "preprocessed",
                "dataset_properties.pkl",
            )
        )
        transforms = AscentPredictor.get_predict_transforms(dataset_properties)
        datalist = AscentPredictor.check_input_folder_and_return_datalist(
            cfg.input_folder,
            cfg.output_folder,
            cfg.overwrite_existing,
            len(dataset_properties["modalities"].keys()),
        )

        dataset = CacheDataset(data=datalist, transform=transforms, cache_rate=1.0)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            shuffle=False,
        )

        log.info("Starting predicting!")
        log.info(f"Using checkpoint: {cfg.ckpt_path}")
        trainer.predict(model=model, dataloaders=dataloader, ckpt_path=cfg.ckpt_path)

        metric_dict = trainer.callback_metrics

        return metric_dict, object_dict


def main():
    """Run the script."""
    AscentPredictor.main()


if __name__ == "__main__":
    main()
