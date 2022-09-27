# author_name = 'Preetham Ganesh'
# project_title = 'POS Tagging-based Neural Machine Translation System for European Languages'
# email = 'preetham.ganesh2015@gmail.com'


import os
import logging
import warnings
import sys


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
warnings.filterwarnings("ignore")


import requests
import zipfile
import tarfile
import pandas as pd


def check_directory_path_existence(directory_path: str) -> str:
    """Creates the absolute path for the directory path given in argument if it does not already exist.

    Args:
        directory_path: A string which contains the directory path that needs to be created if it does not already
            exist.
    
    Returns:
        A string which contains the absolute directory path.
    """
    # Creates the following directory path if it does not exist.
    home_directory_path = os.path.dirname(os.getcwd())
    absolute_directory_path = "{}/{}".format(home_directory_path, directory_path)
    if not os.path.isdir(absolute_directory_path):
        os.makedirs(absolute_directory_path)
    return absolute_directory_path


def create_log(logger_directory_path: str, log_file_name: str) -> None:
    """Creates an object for logging terminal output.

    Args:
        logger_directory_path: A string which contains the location where the log file should be stored.
        log_file_name: A string which contains the name for the log file.
    
    Returns:
        None.
    """
    # Checks if the following path exists.
    logger_directory_path = check_directory_path_existence(logger_directory_path)

    # Create and configure logger
    logging.basicConfig(
        filename="{}/{}".format(logger_directory_path, log_file_name),
        format="%(asctime)s %(message)s",
        filemode="w",
    )
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def log_information(log: str) -> None:
    """Saves current log information, and prints it in terminal.

    Args:
        log: A string which contains the information that needs to be printed in terminal and saved in log.
    
    Returns:
        None.
    
    Exception:
        NameError: When the logger is not defined, this exception is thrown.
    """
    try:
        logger.info(log)
    except NameError:
        _ = ""
    print(log)


def download_file_from_url(url: str, directory_path: str) -> str:
    """Downloads file from the URL provided and stores it locally.

    Args:
        url: A string which contains URL where the file is located on the internet.
        directory_path: A string which contains the path where the file needs to be saved.
    
    Returns:
        A string which contains absolution file path where the downloaded file is stored.
    """
    # Checks if the following directory path exists.
    directory_path = check_directory_path_existence(directory_path)

    # Obtains response for the URL.
    response = requests.get(url, stream=True)

    # Extracts file name from URL.
    file_name = url.rsplit("/")[-1]

    # Downloaded File path.
    file_path = '{}/{}'.format(directory_path, file_name)

    # If the status code for the URL code is OK, then the file is downloaded.
    if response.status_code == 200 or response.status_code == 406:
        with open(file_path, 'wb') as out_file:
            out_file.write(response.raw.read())
        out_file.close()
        log_information("Downloaded file from URL and saved at {} successfully.".format(file_path))
        log_information("")
        return file_path

    else:
        log_information("The URL request produced {} status code.".format(response.status_code))
        log_information('')
        sys.exit()


def extract_from_tar_file(tar_file_path: str, extracted_files_directory_path: str) -> None:
    """Extracts files from the compressed TAR file.

    Args:
        tar_file_path: A string which contains the location where the compressed TAR file is located.
        extracted_files_directory_path: A string which contains the directory path where the extracted files should be 
            saved.
    
    Returns:
        None.
    """
    # Checks if the following directory path exists.
    extracted_files_directory_path = check_directory_path_existence(extracted_files_directory_path)

    # Extracts files from the compressed TAR file.
    tar_file = tarfile.open(tar_file_path)
    tar_file.extractall(extracted_files_directory_path)
    tar_file.close()
