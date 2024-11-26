import os

from utility.load import load_yaml


if __name__ == "__main__":
    # Load the configuration file
    config = load_yaml(os.environ.get('CONFIG_FILE'))