# authors_name = 'Preetham Ganesh, Bharat S Rawal, Alexander Peter, Andi Giri'
# project_title = 'POS Tagging-based Neural Machine Translation System for European Languages using Transformers'
# email = 'preetham.ganesh2015@gmail.com, rawalksh001@gannon.edu, apeter@softsquare.biz, asgiri@softsquare.biz'
# doi = 'www.doi.org/10.37394/23209.2021.18.5'


import os
import sys
import logging

import tensorflow_datasets as tfds
import zipfile
import tarfile


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def download_extract_datasets(european_language: str) -> None:
    """Downloads the paracrawl dataset for the current european language. Extracts text files from manually downloaded
    datasets i.e. manythings and europarl datasets.

    Args:
        european_language: A string which contains the abbreviation for the current European language.

    Returns:
        None.
    """
    manythings_abbreviations = {'de': 'deu', 'es': 'spa', 'fr': 'fra'}
    # Downloads paracrawl dataset into the corresponding directory for the current european language.
    _ = tfds.load('para_crawl/en{}'.format(european_language), split='train', shuffle_files=True,
                  data_dir='../data/downloaded_data/{}-en/paracrawl'.format(european_language))
    print()
    print('Downloaded paracrawl dataset for {}-en'.format(european_language))
    print()
    # Extracts manythings dataset for the current european language.
    with zipfile.ZipFile('../data/downloaded_data/{}-en/{}-eng.zip'.format(
            european_language, manythings_abbreviations[european_language])) as file:
        file.extract('spa.txt', '../data/extracted_data/{}-en/manythings'.format(european_language))
    file.close()
    print('Extracted manythings dataset for {}-en'.format(european_language))
    print()
    # Extracts Europarl dataset for the current european language.
    file = tarfile.open('../data/downloaded_data/{}-en/{}-en.tgz'.format(european_language, european_language))
    file.extractall('../data/extracted_data/{}-en/europarl'.format(european_language))
    file.close()
    print('Extracted Europarl dataset for {}-en'.format(european_language))


def main():
    print()
    european_language = sys.argv[1]
    download_extract_datasets(european_language)
    print()


if __name__ == '__main__':
    main()
