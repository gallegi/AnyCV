import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class SingleTargetDataset(Dataset):
    """Dataset class that load pairs of (image, single classification label).
    
    Attributes:
        df: 
            a Pandas DataFrame contains a 2 required columns 
                'filepath': path to the image
                the other having name from the arg label_col: interger label
        label_col: name of the column containing interger label
        transforms: albumentation transformations that will be applied on the images
    """
    def __init__(self, df:pd.DataFrame, label_col:str='class', transforms=None):
        """See class attributes"""
        self.df = df.reset_index()
        self.label_col = label_col
        self.augmentations = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        """Get data and label by row index of the dataframe df
        
        Args:
            index: the i-th pair of (image, label) to be retrieved

        Returns: a dictionary with 2 keys
            input: image in a form of a tensor
            target (optional): label of the image
        """
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
