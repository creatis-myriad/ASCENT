import numpy as np
from einops.einops import rearrange
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import LoadImage, SqueezeDim, ToTensor
from monai.transforms.transform import MapTransform, Transform

from covid.utils.file_and_folder_operations import load_pickle


class Convert3Dto2D(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = rearrange(d[key], "c w h d -> (c d) w h")
        return d


class Convert2Dto3D(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        num_channel,
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(keys, allow_missing_keys)
        self.num_channel = num_channel

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = rearrange(d[key], "(c d) w h -> c w h d", c=self.num_channel)
        return d


class MayBeSqueezed(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        dim,
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(keys, allow_missing_keys)
        self.squeeze = SqueezeDim(dim=dim)
        self.dim = dim

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if len(d[key].shape) == 4 and d[key].shape[self.dim] == 1:
                d[key] = self.squeeze(d[key])
        return d


class LoadNpyd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        test: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:

        super().__init__(keys, allow_missing_keys)
        self.test = test

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == "data":
                # case_all_data = np.load(d[key], "r")
                case_all_data = LoadImage(image_only=True, channel_dim=0)(d[key])
                meta = case_all_data._meta
                d.pop(key, None)
                d["image"] = MetaTensor(case_all_data.array[:-1].astype(np.float32), meta=meta)
                d["label"] = MetaTensor(case_all_data.array[-1:].astype(np.uint8), meta=meta)
                del case_all_data
                assert len(d["image"].shape) == 4, "Image should be (c, w, h, d)"
                assert len(d["label"].shape) == 4, "Label should be (c, w, h, d)"
            elif key == "image_meta_dict":
                if self.test:
                    image_meta_dict = load_pickle(d["image_meta_dict"])
                    d["image_meta_dict"] = {}
                    d["image_meta_dict"]["case_identifier"] = image_meta_dict["case_identifier"]
                    d["image_meta_dict"]["original_shape"] = image_meta_dict["original_shape"]
                    d["image_meta_dict"]["original_spacing"] = image_meta_dict["original_spacing"]
                    d["image_meta_dict"]["shape_after_cropping"] = image_meta_dict[
                        "shape_after_cropping"
                    ]
                    d["image_meta_dict"]["crop_bbox"] = image_meta_dict["crop_bbox"]
                    d["image_meta_dict"]["resampling_flag"] = image_meta_dict["resampling_flag"]
                    if d["image_meta_dict"]["resampling_flag"]:
                        d["image_meta_dict"]["shape_after_resampling"] = image_meta_dict[
                            "shape_after_resampling"
                        ]
                        d["image_meta_dict"]["anisotrophy_flag"] = image_meta_dict[
                            "anisotrophy_flag"
                        ]
            else:
                raise NotImplementedError
        return d


if __name__ == "__main__":
    data_path = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/CAMUS/preprocessed/data_and_properties/NewCamus_0001.npy"
    # data_path = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/CAMUS/cropped/NewCamus_0001.npz"
    prop = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/CAMUS/preprocessed/data_and_properties/NewCamus_0001.pkl"
    data = LoadNpyd(["data", "image_meta_dict"], test=True)(
        {"data": data_path, "image_meta_dict": prop}
    )
