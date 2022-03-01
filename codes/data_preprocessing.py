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

from utils import load_json_file


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def load_data_subset(european_language: str,
                     dataset_name: str,
                     starting_index: int,
                     ending_index: int) -> tuple:
    """Loads data subset based on the current European language and dataset name. If the dataset name is paracrawl, then
    the starting and ending indexes are used to select a subset from the paracrawl dataset. If the dataset name
    is either manythings or europarl, both the extracted datasets are returned.

    Args:
        european_language: A string which contains the abbreviation for the current European language.
        dataset_name: A string that contains current dataset name.
        starting_index: An integer which contains the index from which the subset should be chosen.
        ending_index: An integer which contains the index until which the subset should be chosen.

    Returns:
        A tuple which contains 2 lists, one for English language sentences, and one for Europarl language sentences.
    """
    english_sentences, european_language_sentences = list(), list()
    if dataset_name == 'paracrawl':
        # Loads the current subset of paracrawl dataset using the starting and ending indexes as a Pandas dataframe.
        original_current_dataset = tfds.as_dataframe(tfds.load(
            'para_crawl/en{}'.format(european_language), split='train[{}:{}]'.format(starting_index, ending_index),
            data_dir='../data/downloaded_data/{}-en/paracrawl'.format(european_language)))
        # Converts the loaded dataframe into list of sentences.
        english_sentences += list(original_current_dataset['en'])
        european_language_sentences += list(original_current_dataset[european_language])
    else:
        manythings_abbreviations = {'de': 'deu', 'es': 'spa', 'fr': 'fra'}
        # Loads the manythings dataset for the current european language, as a Pandas dataframe.
        manythings_dataset = pd.read_csv('../data/extracted_data/{}-en/manythings/{}.txt'.format(
            european_language, manythings_abbreviations[european_language]), sep='\t', encoding='utf-8',
            names=['en', european_language, '_'])
        # Converts the loaded dataframe into a list of sentences.
        english_sentences += list(manythings_dataset['en'])
        european_language_sentences += list(manythings_dataset[european_language])
        # Reads English sentences, from the Europarl dataset.
        file = open('../data/extracted_data/{}-en/europarl/europarl-v7.{}-en.en'.format(
            european_language, european_language))
        english_sentences += file.read().split('\n')
        file.close()
        # Reads European language sentences, from the Europarl dataset.
        file = open('../data/extracted_data/{}-en/europarl/europarl-v7.{}-en.{}'.format(
            european_language, european_language, european_language))
        european_language_sentences += file.read().split('\n')
        file.close()
    return english_sentences, european_language_sentences


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


#if __name__ == '__main__':
#    main()


"""english_sentences, european_language_sentences = load_data_subset('es', 'manythings', 0, 100)
english_sentence = english_sentences[0].decode('utf-8')
print(english_sentences[0])
print(english_sentence)
european_language_sentence = european_language_sentences[0].decode('utf-8')
print(european_language_sentence)
print(european_language_sentences[0])


def cpu_thread_allocation(european_language: str,
                          n_original_examples: int,
                          n_threads: int,
                          n_sentence_pairs_per_dataset) -> None:
    n_datasets = n_original_examples // int(n_sentence_pairs_per_dataset)
    for i in range(0, n_datasets, n_threads):
        thread_dataset_allocation = []
        if i + n_threads <= n_datasets:
            for j in range(n_threads):
                thread_dataset_allocation.append({'process_id': j, 'dataset': i + j, })
        else:
            n_cpu = n_datasets - i
            for j in range(n_cpu):
                thread_dataset_allocation.append({'process_id': j, 'dataset': i + j})"""

english_sentences, european_languages_sentences = load_data_subset('es', 'europarl', 0, 0)
print(english_sentences[-5:])
print(european_languages_sentences[-5:])
