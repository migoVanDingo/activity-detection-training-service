import os
import torch
from dotenv import load_dotenv
from src.utility.video import load_to_tensor_using_cv2
load_dotenv()

class Preprocess:
    
    @staticmethod
    def preprocess_trims(video_directory: str, trim_list_file: str, output_shape: tuple, output_file: str):
        """
        Preprocess trims and save as a .pt file.
        """
        trims = []

        # Load trims from file
        with open(trim_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                video_path, label = line.split(' ')
                video_path = os.path.join(video_directory, video_path)
                label = int(label.rstrip())

                # Preprocess video
                video_tensor = load_to_tensor_using_cv2(oshape=output_shape, data_aug_flag=True)

                # Append processed data
                trims.append((label, video_tensor))

        # Save as .pt file
        torch.save(trims, output_file)
        print(f"Preprocessed data saved to {output_file}")



