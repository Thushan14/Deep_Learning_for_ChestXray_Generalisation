import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class MultiLabelChestXrayDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        disease_columns,
        transform=None
    ):

        self.data = pd.read_csv(csv_file)

        self.image_dir = image_dir
        self.disease_columns = disease_columns
        self.transform = transform


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_name = row["Image Index"]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        labels = row[
            self.disease_columns
        ].values.astype("float32")

        labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

        if self.transform:

            image = self.transform(image)

        return image, labels