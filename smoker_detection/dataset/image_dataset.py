import os
import os.path
import pandas as pd
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional


class ImageDataset(Dataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        label_list = ["notsmoking", "smoking"]
        self.df = self.create_dataframe(root, label_list)
        self.transforms = transforms

    def create_dataframe(self, root, label_list):
        df = pd.DataFrame({"path": [], "label": [], "class_id": []})
        img_list = glob(os.path.join(root, "*.jpg"))
        for img in img_list:
            file_name = os.path.splitext(os.path.basename(img))[0]
            if file_name.startswith(label_list[0]):
                new_data = pd.DataFrame(
                    {"path": [img], "label": [label_list[0]], "class_id": [0]}
                )
            elif file_name.startswith(label_list[1]):
                new_data = pd.DataFrame(
                    {"path": [img], "label": [label_list[1]], "class_id": [1]}
                )
            else:
                continue
            df = pd.concat([df, new_data], ignore_index=True)
        df[["path"]] = df[["path"]].astype(str)
        df[["label"]] = df[["label"]].astype(str)
        df[["class_id"]] = df[["class_id"]].astype(int)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]["path"]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        class_id = self.df.iloc[index]["class_id"]
        return img, class_id
