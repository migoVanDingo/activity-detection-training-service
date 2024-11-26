# logger_config.py
import logging

def setup_logger(log_file: str = 'default.log', logger_name: str = 'main_logger'):
    """
    Sets up a logger that writes to the specified log file.
    
    Parameters:
    - log_file: The file where logs will be written.
    - logger_name: The name of the logger instance.
    
    Returns:
    - A configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    
    # Avoid adding multiple handlers if the logger is reused
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(lineno)d | \n %(message)-20s'
        )
        file_handler.setFormatter(formatter)
        
        # Set logger level and attach handler
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    return logger
