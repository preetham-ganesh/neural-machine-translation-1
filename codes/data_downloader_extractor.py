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
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def download_extract_paracrawl_dataset(european_language: str,
                                       n_datasets: int,
                                       n_sentence_pairs_per_dataset: int) -> None:
    """Downloads the paracrawl dataset for the current european language, and splits the sentences into multiple sub
    datasets for faster processing.

    Args:
        european_language: A string which contains the abbreviation for the current European language.
        n_datasets: An integer which contains the number of sub_datasets in the Europarl and Manythings datasets.
        n_sentence_pairs_per_dataset: An integer which contains number of sentence pairs per dataset.

    Returns:
        None.

    Raises:
        Assertion Error: If i + n_sentence_pairs_per_dataset is greater than total no. of original examples in the
                         dataset.
    """
    # Downloads paracrawl dataset into the corresponding directory for the current european language.
    _, info = tfds.load('para_crawl/en{}'.format(european_language), split='train', with_info=True, shuffle_files=True,
                        data_dir='../data/downloaded_data/{}-en/paracrawl'.format(european_language))
    print()
    print('Downloaded paracrawl dataset for {}-en'.format(european_language))
    print()
    n_original_paracrawl_examples = info.splits['train'].num_examples
    working_directory = '../data/extracted_data/{}-en/splitted_data'.format(european_language)
    # Iterates across sentence pairs in the Paracrawl dataset, and splits the dataset based on
    # n_sentence_pairs_per_dataset.
    for i in range(0, n_original_paracrawl_examples, n_sentence_pairs_per_dataset):
        try:
            current_dataset = tfds.as_dataframe(tfds.load(
                'para_crawl/en{}'.format(european_language),
                split='train[{}:{}]'.format(i, i + n_sentence_pairs_per_dataset),
                data_dir='../data/downloaded_data/{}-en/paracrawl'.format(european_language)))
        except AssertionError:
            current_dataset = tfds.as_dataframe(tfds.load(
                'para_crawl/en{}'.format(european_language), split='train[{}:]'.format(i),
                data_dir='../data/downloaded_data/{}-en/paracrawl'.format(european_language)))
        file_path = '{}/dataset_{}.csv'.format(working_directory, n_datasets)
        current_dataset.to_csv(file_path, index=False)
        print('Paracrawl dataset_{} saved successfully.'.format(n_datasets))
        print()
        n_datasets += 1


def extract_europarl_datasets(european_language: str,
                              n_sentence_pairs_per_dataset: int) -> int:
    """Extracts the Manythings and Europarl datasets for the current european language, and splits the combined
    sentences into multiple sub datasets for faster processing.

    Args:
        european_language: A string which contains the abbreviation for the current European language.
        n_sentence_pairs_per_dataset: An integer which contains number of sentence pairs per dataset.

    Returns:
        An integer which contains the number of sub_datasets in the Europarl and Manythings datasets.
    """
    manythings_abbreviations = {'de': 'deu', 'es': 'spa', 'fr': 'fra'}
    english_language_sentences, european_language_sentences = list(), list()
    # Extracts manythings dataset for the current european language.
    with zipfile.ZipFile('../data/downloaded_data/{}-en/{}-eng.zip'.format(
            european_language, manythings_abbreviations[european_language])) as file:
        file.extract('{}.txt'.format(manythings_abbreviations[european_language]),
                     '../data/extracted_data/{}-en/manythings'.format(european_language))
    file.close()
    print('Extracted manythings dataset for {}-en.'.format(european_language))
    print()
    # Extracts Europarl dataset for the current european language.
    file = tarfile.open('../data/downloaded_data/{}-en/{}-en.tgz'.format(european_language, european_language))
    file.extractall('../data/extracted_data/{}-en/europarl'.format(european_language))
    file.close()
    print('Extracted Europarl dataset for {}-en.'.format(european_language))
    print()
    # Loads the manythings dataset for the current european language, as a Pandas dataframe.
    manythings_dataset = pd.read_csv('../data/extracted_data/{}-en/manythings/{}.txt'.format(
        european_language, manythings_abbreviations[european_language]), sep='\t', encoding='utf-8',
        names=['en', european_language, '_'])
    # Converts the loaded dataframe into a list of sentences.
    english_language_sentences += list(manythings_dataset['en'])
    european_language_sentences += list(manythings_dataset[european_language])
    # Reads English sentences, from the Europarl dataset.
    file = open('../data/extracted_data/{}-en/europarl/europarl-v7.{}-en.en'.format(
        european_language, european_language))
    english_language_sentences += file.read().split('\n')
    file.close()
    # Reads European language sentences, from the Europarl dataset.
    file = open('../data/extracted_data/{}-en/europarl/europarl-v7.{}-en.{}'.format(
        european_language, european_language, european_language))
    european_language_sentences += file.read().split('\n')
    file.close()
    # Creates the following directory path if it does not exist.
    working_directory = '../data/extracted_data/{}-en/splitted_data'.format(european_language)
    if not os.path.isdir(working_directory):
        os.makedirs(working_directory)
    n_datasets = 0
    # Iterates across the sentence pairs in the Europarl and Manythings datasets, and splits the dataset based on
    # n_sentence_pairs_per_dataset.
    for i in range(0, len(english_language_sentences), n_sentence_pairs_per_dataset):
        current_dataset = pd.DataFrame({'en': english_language_sentences[i: i + n_sentence_pairs_per_dataset],
                                        european_language: european_language_sentences[i: i + n_sentence_pairs_per_dataset]})
        file_path = '{}/dataset_{}.csv'.format(working_directory, n_datasets)
        current_dataset.to_csv(file_path, index=False)
        print('Europarl & Manythings dataset_{} saved successfully.'.format(n_datasets))
        print()
        n_datasets += 1
    return n_datasets


def main():
    print()
    european_language = sys.argv[1]
    n_sentence_pairs_per_dataset = int(sys.argv[2])
    n_datasets = extract_europarl_datasets(european_language, n_sentence_pairs_per_dataset)
    download_extract_paracrawl_dataset(european_language, n_datasets, n_sentence_pairs_per_dataset)


if __name__ == '__main__':
    main()
