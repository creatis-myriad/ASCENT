import numpy as np
from einops.einops import rearrange
from monai.config import KeysCollection
from monai.transforms import SqueezeDim, ToTensor
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


class LoadNpy(Transform):
    def __call__(self, data):
        d = dict(data)
        assert "data" in d.keys(), "Data (.npy) must be present."
        assert "properties" in d.keys(), "Data's properties (.pkl) must be present."
        case_all_data = np.load(d["data"][:-4] + ".npy", "r")
        properties = load_pickle(d["properties"])
        image_npy = case_all_data[:-1].astype(np.float32)
        assert len(image_npy.shape) == 4, "Image should be (c, w, h, d)"
        label_npy = case_all_data[-1:].astype(np.uint8)
        assert len(label_npy.shape) == 4, "Label should be (c, w, h, d)"
        return {
            "image": ToTensor()(image_npy),
            "label": ToTensor()(label_npy),
            "image_meta_dict": properties,
        }


if __name__ == "__main__":
    data_path = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/CAMUS/preprocessed/data_and_properties/NewCamus_0001.npy"
    prop = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/CAMUS/preprocessed/data_and_properties/NewCamus_0001.pkl"
    data = LoadNpy()({"data": data_path, "properties": prop})
