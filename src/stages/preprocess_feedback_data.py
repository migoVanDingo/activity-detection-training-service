import os
from classes.preprocess import Preprocess
from utility.file import FileUtils
from utility.load import LoadUtils
from utility.logger_config import setup_logger
from dotenv import load_dotenv
load_dotenv()
system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

if __name__ == '__main__':
    system_logger.info("STAGE: preprocess feedback data")
    print("STAGE: preprocess feedback data")
    config = LoadUtils.load_yaml(os.environ['PARAMS_PATH'])
    
    
    if config['preprocess_feedback_data']['is_feedback']:

        # Process raw predictions list filtering out class_prob = 0
        Preprocess.filter_predictions(predictions_file_path=config['preprocess_feedback_data']['predictions_file_path'], 
                                      output_file_path=config['preprocess_feedback_data']['filtered_predictions_file_path'])
        

        # Generate trims from filtered predictions
        Preprocess.generate_prediction_trims(filtered_predictions_file=config['preprocess_feedback_data']['filtered_predictions_file_path'], video_directory=config['video']['video_dir'], output_directory=config['video']['trims_dir'], output_file_path=config['preprocess_feedback_data']['prediction_trims_file_path'])

        # Merge all trims into a single file
        merge_files = [config['preprocess']['trim_list_file'], config['preprocess_feedback_data']['prediction_trims_file_path'], config['preprocess']['val_data']]
        Preprocess.merge_prediction_dataset(files=merge_files, output_file=config['preprocess_feedback_data']['merged_trims_file'])

        # Split merged trims into train and test
        Preprocess.train_test_split(merged_data_file_path=config['preprocess_feedback_data']['merged_trims_file'], output_train_file=config['training_data']['merge_dataset_file'], output_test_file=config['validation_data']['merge_dataset_file'], test_size=config['validation']['test_size'])


    else:
    
        if FileUtils.check_file_path(config['training_data']['merge_data']) and FileUtils.check_file_path(config['validation_data']['merge_data']):
            system_logger.info("Feedback data processing not required")
            print("Feedback data processing not required")
        else:
            # open preprocess trim_list_file copy it to training data merged dataset file
            FileUtils.copy_file_contents(config['preprocess']['trim_list_file'], config['training_data']['merge_dataset_file'])
            # open preprocess val_data copy it to validation data merged dataset file
            FileUtils.copy_file_contents(config['preprocess']['val_data'], config['validation_data']['merge_dataset_file'])
            
            system_logger.info("Feedback data processing not required, files copied from preprocess data")
            print("Feedback data processing not required, files copied from preprocess data")
        

    system_logger.info("Completed preprocessing feedback data\n")
    print("Completed preprocessing feedback data\n")
            