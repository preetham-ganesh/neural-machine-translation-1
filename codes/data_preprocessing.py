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


def remove_html_markup(sentence: str) -> str:
    """Removes HTML markup from sentences given as input and returns processed sentences.

    Args:
        sentence: Input sentence from which HTML markups should be removed (if it exists).

    Returns:
        Processed sentence from which HTML markups are removed (if it exists).
    """
    tag = False
    quote = False
    processed_sentence = ''
    for i in range(len(sentence)):
        if sentence[i] == '<' and not quote:
            tag = True
        elif sentence[i] == '>' and not quote:
            tag = False
        elif (sentence[i] == '"' or sentence[i] == "'") and tag:
            quote = not quote
        elif not tag:
            processed_sentence += sentence[i]
    return processed_sentence


def main():
    print()
    european_language = sys.argv[1]
    _, original_data_info = tfds.load('para_crawl/en{}'.format(european_language), split='train', with_info=True)
    print()
    print('Datasets loaded successfully.')
    print()
    n_original_examples = original_data_info.splits['train'].num_examples
    n_sentences_pairs_per_dataset = 100000
    n_threads = int(sys.argv[2])
    #cpu_thread_allocation(european_language, n_original_examples, n_threads, n_sentences_pairs_per_dataset)


if __name__ == '__main__':
    main()
