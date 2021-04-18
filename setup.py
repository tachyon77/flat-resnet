"""

Usage:    
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Mohammad Mahbubuzzaman (tachyon77@gmail.com)
"""

import numpy as np
import os
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    #print('Downloading spacy language model...')
    #run(['python', '-m', 'spacy', 'download', 'en'])



def process_file(filename, data_type, word_counter, char_counter):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)        
        for article in tqdm(source["data"]):
            pass            
        print(f"{len(examples)} questions in total")
    return examples, eval_examples


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies

    build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)
    if args.include_test_examples:
        test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter)
        save(args.test_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, "test",
                                   args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

   
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")  
    save(args.dev_meta_file, dev_meta, message="dev meta")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    download(args_)

    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    pre_process(args_)
