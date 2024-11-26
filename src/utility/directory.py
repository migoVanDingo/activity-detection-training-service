import os

class DirectoryUtils:

    @staticmethod
    def check_directory_path(dir_path: str) -> bool:
        """
        Check if directory exists
        :param dir_path :: str : path to directory
        :return: True if directory exists, False otherwise
        """
        return os.path.isdir(dir_path)

    @staticmethod
    def read_directory(dir_path: str) -> list:
        """
        Read directory
        :param dir_path :: str : path to directory
        :return: list of files in directory
        """
        return os.listdir(dir_path)

    @staticmethod
    def create_directory(dir_path: str) -> bool:
        """
        Create directory
        :param dir_path :: str : path to directory
        :return: True if directory created, False otherwise
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            return True
        return False