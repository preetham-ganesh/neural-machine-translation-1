# authors_name = 'Preetham Ganesh, Bharat S Rawal, Alexander Peter, Andi Giri'
# project_title = 'POS Tagging-based Neural Machine Translation System for European Languages using Transformers'
# email = 'preetham.ganesh2015@gmail.com, rawalksh001@gannon.edu, apeter@softsquare.biz, asgiri@softsquare.biz'
# doi = 'www.doi.org/10.37394/23209.2021.18.5'


import os
import re
import sys

import multiprocessing
import pandas as pd
import unicodedata
from sklearn.utils import shuffle


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
                        language: str,
                        sentence_max_length: int) -> str:
    """Pre-processes a sentences based on the language, to remove unwanted characters, lowercase the sentence, etc., and
    returns the processed sentence.

    Args:
        sentence: A string or bytes which contains the input sentence that needs to be processed.
        language: A string which contains the language to which the input sentence belongs to.
        sentence_max_length: An integer which contains the maximum length of a sentence in the dataset.

    Returns:
        The processed sentence that does not have unwanted characters, lowercase letters, and many more.

    Raises:
        AttributeError: If the type of sentence is not bytes, then the error is raised.
    """
    # If the input sentence is of type bytes, it is converted into string by decoding it using UTF-8 format.
    try:
        sentence = sentence.decode('utf-8')
    except AttributeError:
        _ = ''
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
    # If the number of tokens in the sentence is greater than sentence max length, then empty string is returned, else,
    # the processed sentence is returned.
    if len(sentence.split(' ')) > sentence_max_length:
        return ''
    else:
        return sentence


def create_sub_dataset(english_language_sentences: list,
                       european_language_sentences: list,
                       european_language: str,
                       sentence_max_length: int) -> pd.DataFrame:
    """Processes the English and European language sentences, in the current sub-datasset.

    Args:
        english_language_sentences: A list which contains the English language sentences.
        european_language_sentences: A list which contains the European language sentences.
        european_language: A string which contains the abbreviation for the current European language.
        sentence_max_length: An integer which contains the maximum length of a sentence in the dataset.

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
            english_language_processed_sentence = preprocess_sentence(english_language_sentences[i], 'en',
                                                                      sentence_max_length)
            # Pre-processes European language sentence.
            european_language_processed_sentence = preprocess_sentence(european_language_sentences[i],
                                                                       european_language, sentence_max_length)
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


def sub_dataset_preprocessing(current_thread_information: dict) -> None:
    """Using the thread information, loads the sub_dataset, pre-processes the sentence pairs, drops duplicates, drops
    sentence pairs with length greater than 40 tokens, and saves the processed_dataset.

    Args:
        current_thread_information: A dictionary which contains the information for current thread.

    Returns:
        None.
    """
    print('Started processing dataset_{} with thread id {}.'.format(current_thread_information['dataset_number'],
                                                                    current_thread_information['thread_id']))
    print()
    current_sub_dataset = pd.read_csv('../data/extracted_data/{}-en/splitted_data/dataset_{}.csv'.format(
        current_thread_information['european_language'], current_thread_information['dataset_number']))
    english_language_sentences = list(current_sub_dataset['en'])
    european_language_sentences = list(current_sub_dataset[current_thread_information['european_language']])
    print('Loaded sentences from dataset_{} for pre-processing with thread id {}.'.format(
        current_thread_information['dataset_number'], current_thread_information['thread_id']))
    print()
    print('No. of original English language sentences in the dataset_{} with thread id {}: {}'.format(
        current_thread_information['dataset_number'], current_thread_information['thread_id'],
        len(english_language_sentences)))
    print()
    print('No. of original European language ({}) sentences in the dataset_{} with thread id {}: {}'.format(
        current_thread_information['european_language'], current_thread_information['dataset_number'],
        current_thread_information['thread_id'], len(european_language_sentences)))
    print()
    processed_sub_dataset = create_sub_dataset(english_language_sentences, european_language_sentences,
                                               current_thread_information['european_language'],
                                               current_thread_information['sentence_max_length'])
    print('No. of processed sentence pairs in the dataset_{} with thread id {}: {}'.format(
        current_thread_information['dataset_number'], current_thread_information['thread_id'],
        len(processed_sub_dataset)))
    print()
    processed_sub_dataset = drop_duplicates(processed_sub_dataset, current_thread_information['european_language'])
    print('No. of unique processed sentence pairs in the dataset_{} with thread id {}: {}'.format(
        current_thread_information['dataset_number'], current_thread_information['thread_id'],
        len(processed_sub_dataset)))
    print()
    # Creates the following directory path if it does not exist.
    home_directory = os.path.dirname(os.getcwd())
    working_directory = '{}/data/processed_data/{}-en/splitted_data'.format(
        home_directory, current_thread_information['european_language'])
    if not os.path.isdir(working_directory):
        os.makedirs(working_directory)
    file_path = '{}/dataset_{}.csv'.format(working_directory, current_thread_information['dataset_number'])
    processed_sub_dataset.to_csv(file_path, index=False)
    print('Finished processing sentence pairs in dataset_{} with thread id {}, and saved successfully.'.format(
        current_thread_information['dataset_number'], current_thread_information['thread_id']))
    print()


def cpu_thread_allocation(european_language: str,
                          n_threads: int,
                          sentence_max_length: int) -> None:
    """Allocates threads for each sub_dataset iteratively, and performs pre-processing.

    Args:
        european_language: A string which contains the abbreviation for the current European language.
        n_threads: An integer which contains total number of threads allocated for pre-processing the dataset.
        sentence_max_length: An integer which contains the maximum length of a sentence in the dataset.

    Returns:
        None.
    """
    # Identifies the number of datasets, in the specified directory.
    working_directory = '../data/extracted_data/{}-en/splitted_data'.format(european_language)
    n_datasets = len(os.listdir(working_directory))
    # Iterates across all the datasets, stores information for each datasets and allocates thread.
    for i in range(0, n_datasets, n_threads):
        thread_dataset_allocation = []
        if i + n_threads <= n_datasets:
            for j in range(n_threads):
                thread_dataset_allocation.append({'dataset_number': i + j, 'thread_id': j,
                                                  'european_language': european_language,
                                                  'sentence_max_length': sentence_max_length})
            pool = multiprocessing.Pool(processes=n_threads)
        else:
            n_threads = n_datasets - i
            for j in range(n_threads):
                thread_dataset_allocation.append({'dataset_number': i + j, 'thread_id': j,
                                                  'european_language': european_language,
                                                  'sentence_max_length': sentence_max_length})
            pool = multiprocessing.Pool(processes=n_threads)
        pool.map(sub_dataset_preprocessing, thread_dataset_allocation)
        pool.close()


def combine_sub_datasets(european_language: str) -> None:
    """Combines the sub_datasets generated into a single dataset and saves the combined dataset.

    Args:
        european_language: A string which contains the abbreviation for the current European language.

    Returns:
        None.
    """
    working_directory = '../data/processed_data/{}-en/splitted_data'.format(european_language)
    n_datasets = len(os.listdir(working_directory))
    # Creates an empty dataframe for storing sentence pairs from all the sub_datasets.
    combined_dataset = pd.DataFrame(columns=['en', european_language])
    # Iterates across all the saved sub_datasets for combining them.
    for i in range(n_datasets):
        file_path = '{}/dataset_{}.csv'.format(working_directory, i)
        current_sub_dataset = pd.read_csv(file_path)
        combined_dataset = combined_dataset.append(current_sub_dataset, ignore_index=True)
    combined_dataset = shuffle(combined_dataset)
    print('Total no. of sentence pairs in the combined dataset: {}'.format(len(combined_dataset)))
    print()
    combined_dataset = drop_duplicates(combined_dataset, european_language)
    print('Total no. of unique processed sentence pairs in the combined dataset: {}'.format(len(combined_dataset)))
    print()
    file_path = '../data/processed_data/{}-en/train.csv'.format(european_language)
    combined_dataset.to_csv(file_path, index=False)
    print('Finished saving the combined dataset with processed sentences for {}-en.'.format(european_language))
    print()


def main():
    print()
    european_language = sys.argv[1]
    n_threads = int(sys.argv[2])
    sentence_max_length = int(sys.argv[3])
    cpu_thread_allocation(european_language, n_threads, sentence_max_length)
    combine_sub_datasets(european_language)


if __name__ == '__main__':
    main()
