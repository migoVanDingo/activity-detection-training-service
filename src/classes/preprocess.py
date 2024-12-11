import os
import traceback
from typing import IO
import pandas as pd
import torch
from dotenv import load_dotenv
from utility.video import VideoUtils
from utility.logger_config import setup_logger
load_dotenv()
system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

class Preprocess:

    @staticmethod
    def filter_predictions(predictions_file_path: str, output_file_path: str) -> None:
        try:
            # Load csv file into dataframe
            df = pd.read_csv(predictions_file_path)
            
            # Filter predictions on field class_prob != 0
            df = df[df['class_prob'] != 0]

            # Write these to a new file, save file
            df.to_csv(output_file_path, index=False)
            system_logger.info(f"Filtered predictions saved to: {output_file_path}")
        except Exception as e:
            print(f"{__class__.__name__} -- {traceback.format_exc()} -- Error: {e}")
            system_logger.error(f"{__class__.__name__} -- {traceback.format_exc()} - {e} Error: {e}")

    @staticmethod
    def merge_prediction_dataset(files: list, output_file: str) -> None:
        # Load all files into a list
        dfs = [pd.read_csv(file) for file in files]

        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)

        # Save to a new file
        df.to_csv(output_file, index=False)

    @staticmethod
    def train_test_split(merged_data_file_path: str, output_train_file: str, output_test_file: str, test_size: float) -> None:
        try:
            # Load csv file into dataframe
            df = pd.read_csv(merged_data_file_path)
            
            # Split data into train and test
            train_df = df.sample(frac=(1 - test_size), random_state=42)
            test_df = df.drop(train_df.index)

            # Save to new files
            train_df.to_csv(output_train_file, index=False)
            test_df.to_csv(output_test_file, index=False)
            system_logger.info(f"Train data saved to: {output_train_file}")
            system_logger.info(f"Test data saved to: {output_test_file}")
        except Exception as e:
            print(f"{__class__.__name__} -- {traceback.format_exc()} -- Error: {e}")
            system_logger.error(f"{__class__.__name__} -- {traceback.format_exc()} - {e} Error: {e}")

    
    @staticmethod
    def generate_prediction_trims(filtered_predictions_file: str, video_directory: str, output_directory: str, output_file_path: str) -> None:
        try:
            # Load csv file into dataframe
            df = pd.read_csv(filtered_predictions_file)
            output_list = []

            # For each row use video name and look up use the video name, join with video directory and get video properties
            for index, row in df.iterrows():
                video_name = row['video_name']
                video_path = os.path.join(video_directory, video_name)
                video_props = VideoUtils.get_video_properties(video_path)

                # Get the start and end frames
                start_frame = int(row['f0'])
                end_frame = int(row['f1'])

                # Get the bounding box
                bbox = [int(row['w0']), int(row['h0']), int(row['w']), int(row['h'])]

                # Save the spatiotemporal trim
                output_path = os.path.join(output_directory, row['activity'], f"{video_name}_{start_frame}_{end_frame}.mp4")
                VideoUtils.save_spatiotemporal_trim(video_props, start_frame, end_frame, bbox, output_path) 

                # Append to output list for each row write path, space, label 1 if row activity is typing else 0
                output_list.append(f"{row['activity']}/{output_path} {1 if row['activity'] == 'typing' else 0}")

            # Write to file
            with open(os.path.join(output_directory, output_file_path), 'w') as f:
                f.write('\n'.join(output_list))
                f.close()
            
            system_logger.info(f"Generated trims list file: {os.path.join(output_directory, output_file_path)}")
            print(f"{__class__.__name__} -- Generated trims list file: {os.path.join(output_directory, output_file_path)}")
        except FileNotFoundError:
            print(f"{__class__.__name__} -- {traceback.format_exc()} -- Error file not found: {filtered_predictions_file}")
            system_logger.info(f"{__class__.__name__} -- {traceback.format_exc()} -- Error file not found: {filtered_predictions_file}")

        except Exception as e:
            print(f"{__class__.__name__} -- {traceback.format_exc()} -- Error: {e}")
            system_logger.error(f"{__class__.__name__} -- {traceback.format_exc()} - {e} Error: {e}")


        


        



   

    
    @staticmethod
    def load_data_to_tensor(video_directory: str, trim_list_file: str, output_shape: tuple, output_file: str, data_aug_flag: bool) -> None:
        try:
            """
            Preprocess trims and save as a .pt file.
            """
            system_logger.info(f"Preprocessing data from: {trim_list_file}")
            print(f"{__class__.__name__} -- Preprocessing data from: {trim_list_file}")
            trims = []
            

            # Load trims from file
            with open(trim_list_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    video_path, label = line.split(' ')
                    video_path = os.path.join(video_directory, video_path)
                    label = int(label.rstrip())

                    print(f"Processing video: {video_path}")

                    # Preprocess video
                    video_tensor = VideoUtils.load_to_tensor_using_cv2(video_path, oshape=output_shape, data_aug_flag=data_aug_flag)

                    # Append processed data
                    trims.append((label, video_tensor))

            # Save as .pt file
            print(f"===============> Saving preprocessed data to: {output_file}")
            torch.save(trims, output_file)
            system_logger.info(f"Preprocessed data saved to: {output_file}")
            print(f"{__class__.__name__} -- Preprocessed data saved to: {output_file}")
        except FileNotFoundError:
            system_logger.info(f"{__class__.__name__} -- {traceback.format_exc()} -- Error file not found: {trim_list_file}")
        except Exception as e:
            system_logger.error(f"{__class__.__name__} -- {traceback.format_exc()} - {e} Error: {e}")


