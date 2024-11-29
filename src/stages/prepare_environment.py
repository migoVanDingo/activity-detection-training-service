import os

import torch

from utility.constant import Constant
from utility.device import DeviceUtils
from utility.directory import DirectoryUtils
from utility.load import LoadUtils
from utility.logger_config import setup_logger
from utility.model import ModelUtils
from torchinfo import summary
import sys
from dotenv import load_dotenv
load_dotenv()
system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

if __name__ == "__main__":

    system_logger.info("STAGE: prepare_environment")

    # Load the configuration file
    config = LoadUtils.load_yaml(os.environ['PARAMS_PATH'])

    # Create checkpoint directory
    checkpoint_dir = os.path.join(config['prepare_environment']['working_dir'], Constant.CHECKPOINT_DIRECTORY)
    DirectoryUtils.create_directory(checkpoint_dir)
    system_logger.info(f"Checkpoint directory created at {checkpoint_dir}")

    
    #Generate model summary
    with open(config['prepare_environment']['model_summary'], "w") as file:
        
        model = ModelUtils.load_model(config)
        input_data = [config["load_training_data"]["batch_size"], *config["load_training_data"]["input_shape"]]
        model_summary = str(summary(model, input_data=torch.randn(*input_data), batch_dim=config['training']['batch_size'], device=DeviceUtils.get_cuda_device(config['prepare_environment']['cuda_device_id'])))

        # Save model summary to file
        file.write(model_summary)
        file.close()
        
        # Log the model summary file path
        system_logger.info(f"Model summary generated: {config['prepare_environment']['model_summary']}")



