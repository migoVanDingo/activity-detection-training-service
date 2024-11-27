import os

from src.utility.constant import Constant
from src.utility.device import DeviceUtils
from src.utility.directory import DirectoryUtils
from src.utility.load import LoadUtils
from src.utility.logger_config import setup_logger
from src.utility.model import ModelUtils
from torchinfo import summary

system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

if __name__ == "__main__":

    system_logger.info("STAGE: prepare_environment")

    # Load the configuration file
    config = LoadUtils.load_yaml(os.environ.get('CONFIG_FILE'))

    # Create checkpoint directory
    checkpoint_dir = os.path.join(config['prepare_environment']['working_dir'], Constant.CHECKPOINT_DIRECTORY)
    DirectoryUtils.create_directory(checkpoint_dir)
    system_logger.info(f"Checkpoint directory created at {checkpoint_dir}")

    
    #Generate model summary
    with open(config['prepare_environment']['model_summary'], "w") as file:
        
        model = ModelUtils.load_model(config)
        model_summary = str(summary(model, input_data=config['training']['input_shape'], batch_dim=config['training']['batch_size'], device=DeviceUtils.get_cuda_device(config['prepare_environment']['cuda_device'])))

        # Save model summary to file
        file.write(model_summary)
        file.close()
        
        # Log the model summary file path
        system_logger.info(f"Model summary generated: {config['prepare_environment']['model_summary']}")



