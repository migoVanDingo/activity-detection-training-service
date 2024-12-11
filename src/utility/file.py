import os

import torch

class FileUtils:
    def check_file_path(file_path: str) -> bool:
        """
        Check if file exists
        :param file_path :: str : path to file
        :return: True if file exists, False otherwise
        """
        try:
            return os.path.isfile(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return False
        
    def copy_file_contents(src: str, dest: str) -> None:
        """
        Copy file contents from src to dest
        :param src :: str : source file path
        :param dest :: str : destination file path
        """
        with open(src, "r") as src_file:
            data = src_file.read()
            with open(dest, "w") as dest_file:
                dest_file.write(data)
                dest_file.close()
            src_file.close()


    def create_file(file_path: str) -> bool:
        """
        Create file
        :param file_path :: str : path to file
        :return: True if file created, False otherwise
        """
        print(f"Creating file: {file_path}")
        if not os.path.isfile(file_path):
            open(file_path, "w").close()
            return True
        return False

    def read_file(file_path: str) -> str:
        """
        Read file
        :param file_path :: str : path to file
        :return: file content
        """
        with open(file_path, "r") as file:
            return file.read()
        
    
    def create_empty_pt_file(file_path: str) -> bool:
        """
        Create an empty .pt file using torch.save.
        :param file_path: str : Path to the .pt file
        :return: True if the file is created, False otherwise
        """
        print(f"Attempting to create .pt file: {file_path}")
        
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            print(f"Directory does not exist. Creating: {directory}")
            os.makedirs(directory, exist_ok=True)
        
        # Check if the file already exists
        if not os.path.isfile(file_path):
            torch.save({}, file_path)  # Save an empty dictionary
            print(f"Created .pt file at: {file_path}")
            return True
        
        print(f".pt file already exists: {file_path}")
        return False

