import os
from classes.preprocess import Preprocess
from utility.load import load_yaml


if __name__ == '__main__':
    config = load_yaml(os.environ.get('CONFIG_FILE'))
    data = Preprocess.preprocess_trims(video_directory=config['video_directory'], 
                                       trim_list_file=config['trim_list_file'], 
                                       output_shape=config['output_shape'], 
                                       output_file=config['output_file'],
                                       data_aug_flag=config['data_aug_flag'])
    
