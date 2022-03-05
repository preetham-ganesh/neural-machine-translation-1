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
from sklearn.utils import shuffle

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
    english_language_sentences, european_language_sentences = list(), list()
    if dataset_name == 'paracrawl':
        # Loads the current subset of paracrawl dataset using the starting and ending indexes as a Pandas dataframe.
        original_current_dataset = tfds.as_dataframe(tfds.load(
            'para_crawl/en{}'.format(european_language), split='train[{}:{}]'.format(starting_index, ending_index),
            data_dir='../data/downloaded_data/{}-en/paracrawl'.format(european_language)))
        # Converts the loaded dataframe into list of sentences.
        english_language_sentences += list(original_current_dataset['en'])
        european_language_sentences += list(original_current_dataset[european_language])
    else:
        manythings_abbreviations = {'de': 'deu', 'es': 'spa', 'fr': 'fra'}
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
    return english_language_sentences, european_language_sentences


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
    punctuations = list("-!$&(),./%:;?@=_|$€¿¡«»")
    for i in range(len(punctuations)):
        sentence = sentence.replace(punctuations[i], ' ' + punctuations[i] + ' ')
    sentence = sentence.replace('"', ' " ')
    # Removes any space before the start or after the end of the sentence.
    sentence = sentence.strip()
    # Eliminates duplicate whitespaces between individual tokens.
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def create_sub_dataset(english_language_sentences: list,
                       european_language_sentences: list,
                       european_language: str) -> pd.DataFrame:
    """Processes the English and European language sentences, in the current sub-datasset.

    Args:
        english_language_sentences: A list which contains the English language sentences.
        european_language_sentences: A list which contains the European language sentences.
        european_language: A string which contains the abbreviation for the current European language.

    Returns:
        A Pandas dataframe which contains the processed sentences for English language and European language.
    """
    # Creates an empty dataframe for saving the processed sentences.
    processed_sub_dataset = pd.DataFrame(columns=['en', european_language])
    # Iterates across sentences in the English language and European language.
    for i in range(len(english_language_sentences)):
        # If the current English language sentence or European language sentence is null, then both the sentences are
        # skipped.
        if pd.isnull(english_language_sentences[i]) or pd.isnull(european_language_sentences[i]):
            continue
        else:
            # Pre-processes English language sentence.
            english_language_processed_sentence = preprocess_sentence(english_language_sentences[i], 'en')
            # Pre-processes European language sentence.
            european_language_processed_sentence = preprocess_sentence(european_language_sentences[i],
                                                                       european_language)
            # If the processed sentences for English language or European language is empty, then both the sentences are
            # skipped.
            if english_language_processed_sentence == '' or european_language_processed_sentence == '':
                continue
            # Appends the sentence pair, to the processed dataset.
            current_processed_sentence_pair = {'en': english_language_processed_sentence,
                                               european_language: european_language_processed_sentence}
            processed_sub_dataset = processed_sub_dataset.append(current_processed_sentence_pair, ignore_index=True)
    # Shuffles the processed dataset.
    processed_sub_dataset = shuffle(processed_sub_dataset)
    return processed_sub_dataset


def drop_lines_by_length(processed_sub_dataset: pd.DataFrame,
                         european_language: str,
                         english_sentence_max_length: int,
                         european_sentence_max_length: int) -> pd.DataFrame:
    """Drops sentences from the processed dataset if the length of the sentence is greater than the threshold.

    Args:
        processed_sub_dataset: A Pandas dataframe which contains the processed sentences for English language and
                               European language.
        european_language: A string which contains the abbreviation for the current European language.
        english_sentence_max_length: An integer which contains the maximum length of a sentence from English language.
        european_sentence_max_length: An integer which contains the maximum length of a sentence from European language.

    Returns:
        A pandas dataframe which contains processed sentences for English language and European language where the
        length of sentence is less than or equal to 40.
    """
    # Iterates across pair of sentences in the processed dataset.
    for i in range(len(processed_sub_dataset)):
        # If length of sentence is greater than the maximum length then the sentence pair is dropped from the dataset.
        if len(processed_sub_dataset['en'][i].split(' ')) > english_sentence_max_length or len(
                processed_sub_dataset[european_language][i].split(' ')) > european_sentence_max_length:
            processed_sub_dataset = processed_sub_dataset.drop([i])
    return processed_sub_dataset


def drop_duplicates(processed_sub_dataset: pd.DataFrame,
                    european_language: str) -> pd.DataFrame:
    """Drops duplicate sentence pairs from the processed sub_dataset, based on multiple conditions.

    Args:
        processed_sub_dataset: A Pandas dataframe which contains the processed sentences for English language and
                               European language.
        european_language: A string which contains the abbreviation for the current European language.

    Returns:
        A pandas dataframe which contains unique processed sentence pairs for English language and European language.
    """
    # If number of unique sentences in English language is less than in European language, then the duplicates are
    # dropped based on English language.
    if len(processed_sub_dataset['en'].unique()) < len(processed_sub_dataset[european_language].unique()):
        processed_sub_dataset = processed_sub_dataset.drop_duplicates(subset='en', keep='first')
    # If number of unique sentences in English language is greater than in European language, then the duplicates are
    # dropped based on European language.
    elif len(processed_sub_dataset['en'].unique()) > len(processed_sub_dataset[european_language].unique()):
        processed_sub_dataset = processed_sub_dataset.drop_duplicates(subset=european_language, keep='first')
    # If the number of sentences is same for both the languages, then the common duplicates are dropped.
    else:
        processed_sub_dataset = processed_sub_dataset.drop_duplicates()
    return processed_sub_dataset


def dataset_preprocessing(current_thread_information: dict):
    print('Started processing {}_{} dataset with thread id {}.'.format(
        current_thread_information['dataset_name'], current_thread_information['dataset_no'],
        current_thread_information['thread_id']))
    print()
    english_language_sentences, european_language_sentences = load_data_subset(
        current_thread_information['european_language'], current_thread_information['dataset_name'],
        current_thread_information['starting_index'], current_thread_information['ending_index'])
    print('Loaded sentences from {}_{} dataset for pre-processing with thread id {}.'.format(
        current_thread_information['dataset_name'], current_thread_information['dataset_no'],
        current_thread_information['thread_id']))
    print()
    print('No. of original English language sentences in the {}_{} dataset with thread id {}: {}'.format(
        current_thread_information['dataset_name'], current_thread_information['dataset_no'],
        current_thread_information['thread_id'], len(english_language_sentences)))
    print()
    print('No. of original European language ({}) sentences in the {}_{} dataset with thread id {}: {}'.format(
        current_thread_information['european_language'], current_thread_information['dataset_name'],
        current_thread_information['dataset_no'], current_thread_information['thread_id'],
        len(european_language_sentences)))
    print()
    processed_sub_dataset = create_sub_dataset(english_language_sentences, european_language_sentences,
                                               current_thread_information['european_language'])
    print('No. of processed sentence pairs in the {}_{} dataset with thread id {}: {}'.format(
        current_thread_information['dataset_name'], current_thread_information['dataset_no'],
        current_thread_information['thread_id'], len(processed_sub_dataset)))
    print()





def main():
    print()
    european_language = sys.argv[1]
    n_sentences_pairs_per_dataset = 100000
    n_threads = int(sys.argv[2])
    input_sentence_max_length = int(sys.argv[3])
    target_sentence_max_length = int(sys.argv[4])
    cpu_thread_allocation(european_language, n_threads, n_sentences_pairs_per_dataset)


#if __name__ == '__main__':
#    main()


"""english_sentences, european_language_sentences = load_data_subset('es', 'manythings', 0, 100)
english_sentence = english_sentences[0].decode('utf-8')
print(english_sentences[0])
print(english_sentence)
european_language_sentence = european_language_sentences[0].decode('utf-8')
print(european_language_sentence)
print(european_language_sentences[0])


"""def cpu_thread_allocation(european_language: str,
                          n_threads: int,
                          n_sentence_pairs_per_dataset: int) -> None:
    _, original_data_paracrawl_info = tfds.load('para_crawl/en{}'.format(european_language), split='train',
                                                with_info=True)
    print()
    print('Datasets loaded successfully.')
    print()
    n_original_examples = original_data_info.splits['train'].num_examples
    n_sub_datasets_paracrawl = n_original_examples // int(n_sentence_pairs_per_dataset)
    for i in range(0, n_datasets, n_threads):
        thread_dataset_allocation = []
        if i + n_threads <= n_datasets:
            for j in range(n_threads):
                thread_dataset_allocation.append({'process_id': j, 'dataset': i + j, })
        else:
            n_cpu = n_datasets - i
            for j in range(n_cpu):
                thread_dataset_allocation.append({'process_id': j, 'dataset': i + j})