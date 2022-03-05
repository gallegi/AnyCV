import cv2
import torch
from torch.utils.data import Dataset

class SingleTargetDataset(Dataset):
    
    def __init__(self, df, label_col, transforms=None):
        self.df = self.df.reset_index()
        self.label_col = label_col
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out_dict = dict()
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        if(self.label_col in self.df.columns):
            out_dict['target'] = torch.tensor(row[self.label_col])
        out_dict['input'] = image
       
        return out_dict
