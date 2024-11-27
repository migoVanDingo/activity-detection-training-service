import os
from src.classes.preprocess import Preprocess
from src.utility.load import LoadUtils
from src.utility.logger_config import setup_logger
system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

if __name__ == '__main__':
    system_logger.info("STAGE: preprocess_data")
    config = LoadUtils.load_yaml(os.environ.get('CONFIG_FILE'))
    data = Preprocess.preprocess_ground_truth(video_directory=config['video_directory'], 
                                       trim_list_file=config['preprocess']['trim_list_file'], 
                                       output_shape=config['preprocess']['output_shape'], 
                                       output_file=config['preprocess']['output_file_name'],
                                       data_aug_flag=config['preprocess']['data_aug_flag'])
    
