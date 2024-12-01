import os
import torch.optim as optim
import torch.nn as nn

from classes.load_dataset import LoadDataset
from torch.utils.data import DataLoader
from classes.train_and_validate import TrainAndValidate
from utility.model import ModelUtils
from utility.device import DeviceUtils
from utility.load import LoadUtils

from utility.logger_config import setup_logger
system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

if __name__ == '__main__':
    system_logger.info("STAGE: preprocess_data")
    print("STAGE: preprocess_data")
    config = LoadUtils.load_yaml(os.environ['PARAMS_PATH'])

    # Load Training Data
    train_data = LoadDataset(preprocessed_file=config['preprocess']['output_file_name'])
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['load_training_data']['num_workers'])

    # Load Validation Data
    val_data = LoadDataset(preprocessed_file=config['preprocess']['output_val_file_name'])
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['load_training_data']['num_workers_val'])
    
    # Get Cuda device and load into GPU
    device = DeviceUtils.get_cuda_device(config['prepare_environment']['cuda_device_id'])

    # Load the model
    model = ModelUtils.load_model(config)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    # Configure epoch checkpoints
    trainer = TrainAndValidate(config, model, optimizer, criterion, train_loader, val_loader, device)
    trainer.do_process()

  
