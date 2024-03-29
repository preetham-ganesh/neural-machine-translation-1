# author_name = 'Preetham Ganesh'
# project_title = 'POS Tagging-based Neural Machine Translation System for European Languages'
# email = 'preetham.ganesh2015@gmail.com'


import argparse

from utils import create_log
from utils import log_information
from utils import check_directory_path_existence
from utils import download_file_from_url
from utils import extract_from_tar_file
from utils import extract_from_zip_file


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


def download_extract_europarl_dataset(european_language: str, language_abbreviation: str) -> None:
    """Downloads & extracts Europarl dataset for the current european language.

    Args:
        european_language: A string which contains the name of the current european lanuage.
        language_abbreviation: A string which contains the abbreviation for the current european language.
    
    Returns:
        None.
    """
    # Dictionary which contains URLs for Europarl dataset language-wise.
    dataset_urls = {
        'german': 'https://www.statmt.org/europarl/v7/de-en.tgz', 
        'french': 'https://www.statmt.org/europarl/v7/fr-en.tgz',
        'italian': 'https://www.statmt.org/europarl/v7/it-en.tgz',
        'spanish': 'https://www.statmt.org/europarl/v7/es-en.tgz'
    }

    # Downloads the tar file for current european language.
    tar_file_path = download_file_from_url(dataset_urls[european_language], 'data/original/europarl')

    # Checks if the following directory path exists.
    extracted_files_directory_path = 'data/extracted/europarl/{}'.format(language_abbreviation)

    # Extracts Europarl dataset for the current european language.
    extract_from_tar_file(tar_file_path, extracted_files_directory_path)
    log_information("Extracted Europarl dataset for {} language.".format(european_language))
    log_information("")


def extract_manythings_dataset(european_language: str, language_abbreviation: str) -> None:
    """Extracts Manythings dataset for the current european language.

    Args:
        european_language: A string which contains the name of the current european lanuage.
        language_abbreviation: A string which contains the abbreviation for the current european language.
    
    Returns:
        None.
    """
    # Dictionary which contains file names for manythings dataset.
    manythings_dataset_file_names = {
        'german': 'deu-eng.zip',
        'french': 'fra-eng.zip',
        'italian': 'ita-eng.zip',
        'spanish': 'spa-eng.zip'
    }

    # Checks if the following directory path exists.
    manythings_dataset_directory_path = check_directory_path_existence("data/original/manythings")

    # Zip file path.
    zip_file_path = '{}/{}'.format(manythings_dataset_directory_path, manythings_dataset_file_names[european_language])

    # Checks if the following directory path exists.
    extracted_files_directory_path = check_directory_path_existence('data/extracted/manythings/')

    # Extracts Manythings dataset for the current european language.
    extract_from_zip_file(
        zip_file_path, '{}.txt'.format(
            manythings_dataset_file_names[european_language].split('-')[0]
        ), extracted_files_directory_path
    )
    log_information("Extracted Manythings dataset for {} language.".format(european_language))
    log_information("")


def main():
    log_information('')

    # Obtains the arguments.
    european_language = parse_arguments()

    # Dictionary which contains abbreviations for european languages.
    dataset_abbreviations = {
        "german": 'de-en',
        "italian": 'it-en',
        "spanish": 'es-en',
        "french": 'fr-en'
    }

    # Downloaded & extracts Europarl dataset for the current european language.
    download_extract_europarl_dataset(european_language, dataset_abbreviations[european_language])

    # Extracts Manythings dataset for the current european language.
    extract_manythings_dataset(european_language, dataset_abbreviations[european_language])


if __name__ == "__main__":
    major_version = 1
    minor_version = 0
    revision = 0
    global version
    version = "{}.{}.{}".format(major_version, minor_version, revision)
    create_log("logs", "data_downloader_extractor_v{}.log".format(version))
    main()
