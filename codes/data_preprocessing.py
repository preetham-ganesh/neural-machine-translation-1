# authors_name = 'Preetham Ganesh, Bharat S Rawal, Alexander Peter, Andi Giri'
# project_title = 'POS Tagging-based Neural Machine Translation System for European Languages using Transformers'
# email = 'preetham.ganesh2015@gmail.com, rawalksh001@gannon.edu, apeter@softsquare.biz, asgiri@softsquare.biz'
# doi = 'www.doi.org/10.37394/23209.2021.18.5'


import os
import re
import sys
import logging

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import unicodedata

from utils import load_json_file


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def load_data_subset(european_language: str,
                     dataset_name: str,
                     starting_index: int,
                     ending_index: int) -> tuple:
    """Loads data subset based on parameters.

    Loads data subset based on the current European language and dataset name. If the dataset name is paracrawl, then
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


def preprocess_sentence(sentence: str or bytes,
                        language: str) -> str:
    """Pre-processes a sentences based on the language, to remove unwanted characters, lowercase the sentence, etc., and
    returns the processed sentence.

    Args:
        sentence: A string or bytes which contains the input sentence that needs to be processed.
        language: A string which contains the language to which the input sentence belongs to.

    Returns:
        The processed sentence that does not have unwanted characters, lowercase letters, and many more.

    """
    # If the input sentence is of type bytes, it is converted into string by decoding it using UTF-8 format.
    if type(language) != 'str':
        sentence = sentence.decode('utf-8')
    # Lowercases the letters in the sentence, and removes spaces from the beginning and the end of the sentence.
    sentence = sentence.lower().strip()
    # If sentence contains http or https or xml tags, an empty sentence is returned.
    if 'http' in sentence or 'https' in sentence or 'xml' in sentence or '{}' in sentence:
        return ''
    # Removes HTML markups from the sentence.
    sentence = remove_html_markup(sentence)
    # Replaces newline and tabs with empty spaces.
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.replace('\t', ' ')
    # Converts UNICODE characters to ASCII format.
    sentence = ''.join(i for i in unicodedata.normalize('NFD', sentence) if unicodedata.category(i) != 'Mn')
    # Replaces descriptive tokens with their appropriate symbols.
    sentence = sentence.replace('##at##-##at##', '-')
    sentence = sentence.replace('&apos;', "'")
    sentence = sentence.replace('&quot;', '"')
    # Removes noise tokens from sentence.
    noise_tokens = ['&#91;', '&#93;', '&#124;', ' ']
    for i in range(len(noise_tokens)):
        sentence = sentence.replace(noise_tokens[i], ' ')
    # If language is English, then any characters apart from -!$&(),./%:;?@=_|$€a-z0-9 are filtered out.
    if language == 'en':
        sentence = re.sub(r"[^-!$&(),./%:;?@=_|$€a-z0-9]+", ' ', sentence)
    # If language is Spanish, then any characters apart from -!$&(),./%:;?@=_|$€a-z0-9áéíñóúü¿¡ are filtered out.
    elif language == 'es':
        sentence = re.sub(r"[^-!$&(),./%:;?@=_|$€a-z0-9áéíñóúü¿¡]+", ' ', sentence)
    # If language is French, then any characters apart from -!$&(),./%:;?@=_|$€a-z0-9ùûüÿ€àâæçéèêëïîôœ«» are filtered out.
    elif language == 'fr':
        sentence = re.sub(r"[^-!$&(),./%:;?@=_|$€a-z0-9ùûüÿ€àâæçéèêëïîôœ«»]+", ' ', sentence)
    # If language is German, then any characters apart from -!$&(),./%:;?@=_|$€a-z0-9äöüß are filtered out.
    elif language == 'de':
        sentence = re.sub(r"[^-!$&(),./%:;?@=_|$€a-z0-9äöüß]+", ' ', sentence)
    # If there are consecutive full-stops (.), then it is replaced with a single full-stop.
    sentence = re.sub('\.{2,}', '.', sentence)
    # Converts strings such as 8th, 1st, 3rd, & 2nd, into 8 th, 1 st, 3 rd, & 2 nd.
    sentence = re.sub(r'(\d)th', r'\1 th', sentence, flags=re.I)
    sentence = re.sub(r'(\d)st', r'\1 st', sentence, flags=re.I)
    sentence = re.sub(r'(\d)rd', r'\1 rd', sentence, flags=re.I)
    sentence = re.sub(r'(\d)nd', r'\1 nd', sentence, flags=re.I)
    # Adds space between punctuation tokens.
    punctuations = list("-!$&(),./%:;?€'")
    for i in range(len(punctuations)):
        sentence = sentence.replace(punctuations[i], ' ' + punctuations[i] + ' ')
    sentence = sentence.replace('"', ' " ')
    # Removes any space before the start or after the end of the sentence.
    sentence = sentence.strip()
    # Eliminates duplicate whitespaces between individual tokens.
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


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
    cpu_thread_allocation(european_language, n_original_examples, n_threads, n_sentences_pairs_per_dataset)


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