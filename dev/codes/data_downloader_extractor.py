# author_name = 'Preetham Ganesh'
# project_title = 'POS Tagging-based Neural Machine Translation System for European Languages'
# email = 'preetham.ganesh2015@gmail.com'


import argparse


from utils import create_log
from utils import log_information
import requests


def parse_arguments() -> str:
    """Parses the command line arguments provided during file execution.

    Args:
        None.
    
    Returns:
        A string which contains the name of the european language for which the dataset, will be downloaded.
    """
    # Creates an object for parser.
    parser = argparse.ArgumentParser()

    # Adds argument to the parser.
    parser.add_argument("--european_language", type=str, required=True)

    # Parses the argument.
    args = parser.parse_args()
    return args.european_language


def main():
    log_information('')

    european_language = parse_arguments()



if __name__ == "__main__":
    major_version = 1
    minor_version = 0
    revision = 0
    global version
    version = "{}.{}.{}".format(major_version, minor_version, revision)
    create_log("logs", "data_downloader_extractor_v{}.log".format(version))
    main()
