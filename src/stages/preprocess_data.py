import os
from classes.preprocess import Preprocess
from utility.load import LoadUtils
from utility.logger_config import setup_logger
from dotenv import load_dotenv
load_dotenv()
system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

if __name__ == '__main__':
    system_logger.info("STAGE: preprocess_data")
    print("STAGE: preprocess_data")
    config = LoadUtils.load_yaml(os.environ['PARAMS_PATH'])
    data = Preprocess.preprocess_ground_truth(video_directory=config['preprocess']['video_dir'], 
                                       trim_list_file=config['preprocess']['trim_list_file'], 
                                       output_shape=config['preprocess']['output_shape'], 
                                       output_file=config['preprocess']['output_file_name'],
                                       data_aug_flag=config['preprocess']['data_aug_flag'])
    
