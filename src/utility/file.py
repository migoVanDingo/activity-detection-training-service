import os

class FileUtils:
    def check_file_path(file_path: str) -> bool:
        """
        Check if file exists
        :param file_path :: str : path to file
        :return: True if file exists, False otherwise
        """
        return os.path.isfile(file_path)

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

