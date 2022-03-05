import cv2
from commons.datasets import SingleTargetDataset

class MetricLearningDataset(SingleTargetDataset):
    
    def __init__(self, df, label_col, transforms=None):
        super.__init__(df, label_col, transforms)