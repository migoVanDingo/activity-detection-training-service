import os
import traceback
import torch
from dotenv import load_dotenv
from utility.file import FileUtils
from utility.video import load_to_tensor_using_cv2
from utility.logger_config import setup_logger
load_dotenv()
system_logger = setup_logger(os.environ['SYSTEM_LOG_FILE'], 'main_logger')

class Preprocess:
    
    @staticmethod
    def preprocess_ground_truth(video_directory: str, trim_list_file: str, output_shape: tuple, output_file: str, data_aug_flag: bool):
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
                    video_tensor = load_to_tensor_using_cv2(video_path, oshape=output_shape, data_aug_flag=data_aug_flag)

                    # Append processed data
                    trims.append((label, video_tensor))

            # Save as .pt file
            FileUtils.create_file(os.path.dirname(output_file))
            torch.save(trims, output_file)
            system_logger.info(f"Preprocessed data saved to: {output_file}")
            print(f"{__class__.__name__} -- Preprocessed data saved to: {output_file}")
        except FileNotFoundError:
            system_logger.info(f"{__class__.__name__} -- {traceback.format_exc()} -- Error file not found: {trim_list_file}")
        except Exception as e:
            system_logger.error(f"{__class__.__name__} -- {traceback.format_exc()} - {e} Error: {e}")


