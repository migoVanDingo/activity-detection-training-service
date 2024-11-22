import os, ast
import numpy as np
from torch.utils.data import Dataset
from typing import IO
from dotenv import load_dotenv
load_dotenv()
from src.utility.video import load_to_tensor_using_cv2


class LoadTrimsData(Dataset):
    def __init__(self, video_directory: str, trim_list_file: IO, output_shape: tuple):
        self.video_directory = video_directory
        self.trim_list_file = trim_list_file
        self.output_shape = output_shape
        self.trims = self._load_trims()
    
    def __len__():
        pass

    def __getitem__(self, idx):
        path = self.trims[idx][0]
        label = np.array(int(self.trims[idx][1]))

        # Video tensor
        video_tensor = self._load_video(path)

        return (label, video_tensor)

    def _load_trims(self, path: str):
        """ 
        Read trims list file and create tuple of path and label
        [(<trm path>, <trm label>), ...]

        fpth: str
            Text file having video relative path and activity label.
        """
        if os.path.isfile(path):
            with open(path, 'r') as f:
                lines = f.readlines()
                trims = [(f"{self.video_directory}/{x.split(' ')[0]}",
                        x.split(' ')[1].rstrip()) for x in lines]
        return trims
    
    def _load_video(self, path: str):
        """ Load video as tensor using OpenCV video object.
        
        Parameters
        ----------
        vpth: str
            Path to trimmed video
        """
        return load_to_tensor_using_cv2(oshape=self.output_shape, data_aug_flag = True)
        