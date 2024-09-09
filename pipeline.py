import os
import SimpleITK as sitk
import numpy as np
from einops import rearrange
from scipy import stats

from ascent.postprocessing.postprocessing import SegPostprocessor
from ascent.utils.file_and_folder_operations import subfiles

if __name__ == "__main__":
    """Process the whole pipeline for the MYOSAIQ challenge"""

    do_classif = True
    do_mean_mean = True

    classif_path = [
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/0/inference_raw",
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/1/inference_raw",
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/2/inference_raw",
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/3/inference_raw",
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/4/inference_raw"
    ]

    model = ['myosaiq_2d_1stage', 'myosaiq_2d_2stage', 'myosaiq_2d_3stage']
    dataset = 'MYOSAIQ'
    ckpt_path = [
        [
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-07-08_20-05-53/checkpoints/best_epoch_909.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_1/runs/2024-07-09_15-32-52/checkpoints/best_epoch_814.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_2/runs/2024-07-09_15-32-52/checkpoints/best_epoch_669.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_3/runs/2024-07-09_15-34-00/checkpoints/best_epoch_745.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_4/runs/2024-07-09_15-34-42/checkpoints/best_epoch_949.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_5/runs/2024-07-09_15-35-39/checkpoints/best_epoch_689.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_6/runs/2024-07-09_15-37-00/checkpoints/best_epoch_935.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_7/runs/2024-07-09_15-38-03/checkpoints/best_epoch_663.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_8/runs/2024-07-09_15-38-39/checkpoints/best_epoch_572.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_9/runs/2024-07-09_15-39-26/checkpoints/best_epoch_946.ckpt"],
        [
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-07-15_02-51-18/checkpoints/best_epoch_851.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_1/runs/2024-07-15_02-51-40/checkpoints/best_epoch_746.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_2/runs/2024-07-15_02-52-37/checkpoints/best_epoch_864.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_3/runs/2024-07-15_02-53-22/checkpoints/best_epoch_994.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_4/runs/2024-07-15_02-55-09/checkpoints/best_epoch_305.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_5/runs/2024-07-15_03-04-14/checkpoints/best_epoch_972.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_6/runs/2024-07-15_03-04-50/checkpoints/best_epoch_919.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_7/runs/2024-07-15_03-06-11/checkpoints/best_epoch_992.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_8/runs/2024-07-15_03-06-44/checkpoints/best_epoch_744.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_9/runs/2024-07-15_03-07-21/checkpoints/best_epoch_996.ckpt"],
        #[
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-07-20_22-33-21/checkpoints/best_epoch_940.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_1/runs/2024-07-20_22-33-53/checkpoints/best_epoch_963.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_2/runs/2024-07-20_22-34-58/checkpoints/best_epoch_711.ckpt",
        #   "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_3/runs/2024-07-20_22-35-55/checkpoints/best_epoch_990.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_4/runs/2024-07-20_22-36-55/checkpoints/best_epoch_317.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_5/runs/2024-07-20_22-37-56/checkpoints/best_epoch_644.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_6/runs/2024-07-20_22-38-57/checkpoints/best_epoch_802.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_7/runs/2024-07-20_22-39-57/checkpoints/best_epoch_972.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_8/runs/2024-07-20_22-40-58/checkpoints/best_epoch_969.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_9/runs/2024-07-20_22-41-58/checkpoints/best_epoch_902.ckpt"],
        #[
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-07-24_22-09-37/checkpoints/best_epoch_916.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_1/runs/2024-07-24_22-11-07/checkpoints/best_epoch_884.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_2/runs/2024-07-24_22-11-32/checkpoints/best_epoch_888.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_3/runs/2024-07-24_22-12-34/checkpoints/best_epoch_168.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_4/runs/2024-07-24_22-13-08/checkpoints/best_epoch_952.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_5/runs/2024-07-24_22-13-40/checkpoints/best_epoch_996.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_6/runs/2024-07-24_22-14-43/checkpoints/best_epoch_798.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_7/runs/2024-07-24_22-15-12/checkpoints/best_epoch_975.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_8/runs/2024-07-24_22-15-39/checkpoints/best_epoch_851.ckpt",
        #    "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_9/runs/2024-07-24_22-17-23/checkpoints/best_epoch_596.ckpt"],
        [
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-07-30_01-36-14/checkpoints/best_epoch_971.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_1/runs/2024-08-01_15-39-40/checkpoints/best_epoch_883.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_2/runs/2024-08-01_15-50-51/checkpoints/best_epoch_312.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_3/runs/2024-08-01_19-33-25/checkpoints/best_epoch_947.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_4/runs/2024-08-01_19-33-25/checkpoints/best_epoch_742.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_5/runs/2024-08-01_19-35-41/checkpoints/best_epoch_972.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_6/runs/2024-08-01_19-38-42/checkpoints/best_epoch_593.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_7/runs/2024-08-01_20-11-49/checkpoints/best_epoch_890.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_8/runs/2024-08-01_20-15-27/checkpoints/best_epoch_963.ckpt",
            "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_9/runs/2024-08-01_20-29-19/checkpoints/best_epoch_998.ckpt"],
    ]

    input_folder = "data/MYOSAIQ/raw/imagesTs"

    input_label_folder = [None,
                          f"data/MYOSAIQ/raw/labelsTr_mean_stage1_classif{do_classif}",
                          f"data/MYOSAIQ/raw/labelsTr_mean_mean_stage2_classif{do_classif}"]

    for j in range(len(model)):
        for i in range(len(ckpt_path[j])):
            if j != 0:
                output_folder = f"./inference/stage{j + 1}/classif{do_classif}/mean_{i}/"
            else:
                output_folder = f"./inference/stage{j + 1}/{i}/"

            if not os.path.isdir(output_folder):
                print(f"Get results stage {j+1}, model number {i}")
                command = f"ascent_predict dataset={dataset} model={model[j]} model.level={j+1} input_folder={input_folder} input_label_folder={input_label_folder[j]} output_folder={output_folder} ckpt_path={ckpt_path[j][i]} save_npz=True"
                os.system(command)

        files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
        case_ids = np.unique([i[:-12] for i in files if ((i[:-12].endswith("D8")) or (j != 2))])

        if j != 0:
            output_mean = f"./inference/stage{j + 1}/classif{do_classif}/mean/"
        else:
            output_mean = f"./inference/stage{j + 1}/mean/"

        if not os.path.isdir(output_mean):
            os.makedirs(output_mean, exist_ok=True)
            for case in case_ids:
                mask_mean = []
                for i in range(len(ckpt_path)):
                    if j != 0:
                        path_file = f"./inference/stage{j + 1}/classif{do_classif}/mean_{i}/inference_raw/{case}"
                    else:
                        path_file = f"./inference/stage{j + 1}/{i}/inference_raw/{case}"

                    file_nii = sitk.ReadImage(f"{path_file}.nii.gz")
                    spacing = file_nii.GetSpacing()
                    file_npz = np.load(f"{path_file}.npz")['softmax']

                    mask_mean.append(file_npz)

                mask_mean = np.array(mask_mean)
                mask_mean = np.mean(mask_mean, axis=0)

                mask_mean_final = mask_mean.argmax(0)

                if do_mean_mean:
                    np.savez_compressed(os.path.join(output_mean, f"{case}.npz"), softmax=mask_mean)
                    mask_mean_mean = []
                    mask_mean_mean.append(mask_mean)

                    if j != 0:
                        output_mean_mean = f"{output_mean}mean_mean/"
                        os.makedirs(output_mean_mean, exist_ok=True)

                        for k in range(j):
                            if k != 0:
                                path_file = f"./inference/stage{k + 1}/classif{do_classif}/mean/"
                            else:
                                path_file = f"./inference/stage{k + 1}/mean/"

                            file_mean_k = np.load(f"{path_file}{case}.npz")['softmax']
                            mask_mean_k = np.full_like(mask_mean, np.nan)
                            mask_mean_k[:len(file_mean_k)] = file_mean_k

                            mask_mean_mean.append(mask_mean_k)

                        mask_mean_mean = np.array(mask_mean_mean)
                        mask_mean_mean = np.nanmean(mask_mean_mean, axis=0)
                        mask_mean_mean = mask_mean_mean.argmax(0)

                        if j == 1:
                            mask_mean_mean[mask_mean_final == 3] = 3
                        if j == 2:
                            mask_mean_mean[mask_mean_final == 4] = 4

                        itk_image = sitk.GetImageFromArray(rearrange(mask_mean_mean.astype(np.uint8), "w h d ->  d h w"))
                        itk_image.SetSpacing(spacing)
                        sitk.WriteImage(itk_image, os.path.join(output_mean_mean, f"{case}.nii.gz"))

                        postProcessor_mean = SegPostprocessor(output_mean_mean)
                        postProcessor_mean.main(classif_path, do_classif=do_classif)

                mask_mean_final = mask_mean_final.astype(np.uint8)
                itk_image = sitk.GetImageFromArray(rearrange(mask_mean_final, "w h d ->  d h w"))
                itk_image.SetSpacing(spacing)
                sitk.WriteImage(itk_image, os.path.join(output_mean, f"{case}.nii.gz"))

        postProcessor_mean = SegPostprocessor(output_mean)
        postProcessor_mean.main(classif_path, do_classif=do_classif)

        for file in subfiles(f"{output_mean}clean_classif{do_classif}", suffix=".nii.gz"):
            case_identifier = os.path.split(file)[-1].split(".nii.gz")[0]

            image = sitk.ReadImage(file)

            os.makedirs(f"data/MYOSAIQ/raw/labelsTr_mean_stage{j+1}_classif{do_classif}", exist_ok=True)
            sitk.WriteImage(image, os.path.join(f"data/MYOSAIQ/raw/labelsTr_mean_stage{j+1}_classif{do_classif}", f"{case_identifier}.nii.gz"))

    """input_label_folder = [None,
                          f"data/MYOSAIQ/raw/labelsTr_max_voting_stage1_classif{do_classif}",
                          f"data/MYOSAIQ/raw/labelsTr_max_voting_stage2_classif{do_classif}"]

    for j in range(len(model)):
        for i in range(len(ckpt_path[j])):
            if j != 0:
                output_folder = f"./inference/stage{j + 1}/classif{do_classif}/max_voting_{i}/"
            else:
                output_folder = f"./inference/stage{j + 1}/{i}/"

            if not os.path.isdir(output_folder):
                command = f"ascent_predict dataset={dataset} model={model[j]} input_folder={input_folder} input_label_folder={input_label_folder[j]} output_folder={output_folder} ckpt_path={ckpt_path[j][i]} save_npz=True"
                os.system(command)

        files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
        case_ids = np.unique([i[:-12] for i in files])

        if j != 0:
            output_max_voting = f"./inference/stage{j + 1}/classif{do_classif}/max_voting/"
        else:
            output_max_voting = f"./inference/stage{j + 1}/max_voting/"

        if not os.path.isdir(output_max_voting):
            os.makedirs(output_max_voting, exist_ok=True)

            for case in case_ids:
                mask_max_voting = []
                for i in range(len(ckpt_path)):
                    if j != 0:
                        path_file = f"./inference/stage{j + 1}/classif{do_classif}/max_voting_{i}/inference_raw/{case}.nii.gz"
                    else:
                        path_file = f"./inference/stage{j + 1}/{i}/inference_raw/{case}.nii.gz"

                    file_nii = sitk.ReadImage(path_file)
                    spacing = file_nii.GetSpacing()
                    origin = file_nii.GetOrigin()
                    direction = file_nii.GetDirection()

                    mask_max_voting.append(rearrange(sitk.GetArrayFromImage(file_nii), "d h w -> w h d"))

                mask_max_voting = np.array(mask_max_voting)

                mode_result = stats.mode(mask_max_voting, axis=0)
                mask_max_voting = mode_result.mode

                itk_image = sitk.GetImageFromArray(rearrange(mask_max_voting.astype(np.uint8), "w h d ->  d h w"))
                itk_image.SetSpacing(spacing)
                sitk.WriteImage(itk_image, os.path.join(output_max_voting, case + ".nii.gz"))

        postProcessor_max_voting = SegPostprocessor(output_max_voting)
        postProcessor_max_voting.main(classif_path, do_classif=do_classif)

        for file in subfiles(f"{output_max_voting}clean_classif{do_classif}", suffix=".nii.gz"):
            case_identifier = os.path.split(file)[-1].split(".nii.gz")[0]

            image = sitk.ReadImage(file)
            os.makedirs(f"data/MYOSAIQ/raw/labelsTr_max_voting_stage{j+1}_classif{do_classif}", exist_ok=True)
            sitk.WriteImage(image, os.path.join(f"data/MYOSAIQ/raw/labelsTr_max_voting_stage{j+1}_classif{do_classif}", f"{case_identifier}.nii.gz"))"""

    """do_classif = True

    input_folder = "data/MYOSAIQ/raw/imagesTr_t"
    input_label_folder = f"data/MYOSAIQ/raw/labelsTr_mean_stage2_classif{do_classif}"
    output_folder = f"./inf/stage3/classif{do_classif}/mean_0/"

    ckpt_path = "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-07-30_01-38-44/checkpoints/best_epoch_672.ckpt"
    ckpt_path = "C:/Users/goujat/Documents/thesis/ASCENT/logs/MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-08-01_10-27-58/checkpoints/best_epoch_928.ckpt"
    model = 'myosaiq_2d_3stage'
    dataset = 'MYOSAIQ'

    command = f"ascent_predict dataset={dataset} model={model} model.level=3 input_folder={input_folder} input_label_folder={input_label_folder} output_folder={output_folder} ckpt_path={ckpt_path} save_npz=True"
    os.system(command)"""


