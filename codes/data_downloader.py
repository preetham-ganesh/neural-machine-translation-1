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


def main():
    print()
    european_language = sys.argv[1]
    original_train_data = download_dataset(european_language)


if __name__ == '__main__':
    main()
