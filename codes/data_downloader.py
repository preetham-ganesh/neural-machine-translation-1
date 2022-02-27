# authors_name = 'Preetham Ganesh, Bharat S Rawal, Alexander Peter, Andi Giri'
# project_title = 'POS Tagging-based Neural Machine Translation System for European Languages using Transformers'
# email = 'preetham.ganesh2015@gmail.com, rawalksh001@gannon.edu, apeter@softsquare.biz, asgiri@softsquare.biz'
# doi = 'www.doi.org/10.37394/23209.2021.18.5'


import os
import sys
import logging

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def download_dataset(european_language: str) -> tf.data.Dataset:
    """Downloads paracrawl dataset for the current european language.

    Args:
        european_language: A string which contains the abbreviation for the current european language.

    Returns:
        Downloaded paracrawl dataset for the current european language as Tensorflow dataset with numpy arrays.
    """
    original_train_data = tfds.as_numpy(tfds.load('para_crawl/en{}'.format(european_language), split='train',
                                                  shuffle_files=True))
    print()
    print('Datasets loaded successfully.')
    print()
    return original_train_data


def dataset_save(dataset_sentences: pd.DataFrame,
                 working_directory: str,
                 dataset_name: int) -> None:
    """Saves current pairs of sentences if the number of sentences is equal to 1,000,000 or if any remaining pairs are
    available.

    Args:
        dataset_sentences: A pandas dataframe which contains the pair of sentences.
        working_directory: String which contains the path of the directory where the dataset has to be saved.
        dataset_name: Integer which contains the current name of the dataset.

    Returns:
        None.
    """
    file_path = '{}/paracrawl_dataset_{}.csv'.format(working_directory, dataset_name)
    dataset_sentences.to_csv(file_path, index=False)
    print('No. of sentences in the paracrawl_dataset_{}: {}'.format(dataset_name, len(dataset_sentences)))
    print('paracrawl_dataset_{} saved successfully.'.format(dataset_name))
    print()


def convert_tensor_to_sentences(original_train_data: tf.data.Dataset,
                                european_language: str) -> None:
    current_dataset_sentences = pd.DataFrame(columns=['pair_number', 'en', european_language])
    current_index = 0
    current_dataset_name = 0
    home_directory = os.path.dirname((os.getcwd()))
    working_directory = '{}/data/original_data/{}-en/original_data/paracrawl'.format(home_directory, european_language)
    n_sentences_per_file = 1000000
    os.makedirs(working_directory)
    for i in original_train_data:
        english_sentence = str(i['en'].decode('utf-8'))
        european_language_sentence = str(i[european_language].decode('utf-8'))
        current_sentence_pair = {'pair_number': current_index, 'en': english_sentence,
                                 european_language: european_language_sentence}
        current_dataset_sentences = current_dataset_sentences.append(current_sentence_pair, ignore_index=True)
        current_index += 1



def main():
    print()
    european_language = sys.argv[1]
    original_train_data = download_dataset(european_language)


if __name__ == '__main__':
    main()
