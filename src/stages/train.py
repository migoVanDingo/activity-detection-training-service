import os

from classes.load_dataset import LoadDataset
from torch.utils.data import DataLoader
from utility.load import LoadUtils, load_yaml


if __name__ == '__main__':
    config = LoadUtils.load_yaml(os.environ['PARAMS_PATH'])
    load_data = LoadDataset(video_directory=config['data_file'])
    loader = DataLoader(load_data, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    # Get Cuda device and load into GPU



    # Configure epoch checkpoints

  
