from typing import Literal, Sequence

import torch
from monai.transforms import CenterSpatialCropd, SpatialPadd
from torch import Tensor, nn
from torch.nn import Conv2d

from ascent.utils.softmax import softmax_helper
from ascent.utils.transforms import LoadNpyd

if __name__ == "__main__":
    import hydra
    import pyrootutils
    from hydra import compose, initialize
    from matplotlib import pyplot as plt
    from omegaconf import OmegaConf

    from ascent.utils.visualization import dopplermap, imagesc

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    ckpt_path = "C:/Users/ling/Desktop/pdnet/ckpt/ori_best.ckpt"
    data_path = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/DEALIASM/preprocessed/data_and_properties/Dealias_0022.npy"
    cmap = dopplermap(256)
    frame = 6
    # frame = 14
    # frame = 29

    # context initialization
    with initialize(version_base="1.2", config_path="../../../configs/model", job_name="plot"):
        # cfg = compose(config_name="pdnet_2d")
        cfg = compose(config_name="pdnet_2d")
        cfg.scheduler.T_max = 2
        # cfg.net.variant = 2
        cfg.net.iterations = 10
        print(OmegaConf.to_yaml(cfg))

    # instantiate pdnet
    model = hydra.utils.instantiate(cfg)

    # load model weights
    model.load_state_dict(torch.load(ckpt_path, map_location=model.device)["state_dict"])

    pdnet = model.net

    pdnet.eval()

    # load a data
    data = LoadNpyd(["data"], test=False, seg_label=False)({"data": data_path})
    data = SpatialPadd(["image", "label"], spatial_size=[40, 192, -1], mode="constant", value=0)(
        data
    )
    data = CenterSpatialCropd(["image", "label"], roi_size=[40, 192, -1])(data)

    image = data["image"][..., frame][None, :]
    label = data["label"][..., frame][None, :]

    out, results = pdnet.debug(image)
    out = softmax_helper(out)
    out = out.squeeze(0).cpu().detach().numpy().argmax(0)

    plt.figure("Visualization", (18, 6))
    ax = plt.subplot(1, 3, 1)
    imagesc(ax, image[0, 0, ...].transpose(1, 0), "aliased", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 3, 2)
    imagesc(ax, out.transpose(), "prediction", cmap, clim=[0, 2])
    ax = plt.subplot(1, 3, 3)
    imagesc(ax, label[0, 0, ...].transpose(1, 0), "gt", cmap, clim=[0, 2])
    plt.show()

    it = 1

    for result in results:
        fig = plt.figure("Primal Dual", (18, 6))
        # fig.suptitle(f"Iter {it}")
        ax = plt.subplot(1, 3, 1)
        imagesc(ax, (result[0][0, 0, ...]).transpose(1, 0), "f_2", cmap)
        ax = plt.subplot(1, 3, 2)
        imagesc(ax, (result[1][0, 0, ...]).transpose(1, 0), "f_1", cmap)
        ax = plt.subplot(1, 3, 3)
        imagesc(ax, (result[2][0, 0, ...]).transpose(1, 0), "h_1", cmap)
        plt.show()
        it += 1
