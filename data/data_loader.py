from torch.utils.data import Dataset
import torch
from PIL import Image, ImageOps

class DareDataset(Dataset):
    def __init__(self, root_path, mode, transform=None):

        self.root_path = root_path
        self.mode = mode
        self.transform = transform
        self.data = self.read_dataset() # TODO read dataset file via a function

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB') # TODO updated based on the data of each line
        image = ImageOps.exif_transpose(image)

        caption = self.data[idx] # TODO updated based on the data of each line

        return {'image':image, 'caption':caption}

    def read_dataset(self):
        return []
