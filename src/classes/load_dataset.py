import torch
from torch.utils.data import Dataset
from dotenv import load_dotenv
load_dotenv()


class LoadDataset(Dataset):
    def __init__(self, preprocessed_file: str):
        """
        Initialize the dataset from the preprocessed file. File generated from preprocess stage. 
        """
        self.data = torch.load(preprocessed_file)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, video_tensor = self.data[idx]
        return video_tensor, label

        