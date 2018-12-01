from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.utility.plot_utils import plot_and_save_history
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
from keras_text_summarization.library.applications.fake_news_loader import fit_text
import numpy as np

LOAD_EXISTING_WEIGHTS = False


def main():
    np.random.seed(42)
    data_dir_path = './data'
    report_dir_path = './reports'
    model_dir_path = './models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/news_summary.csv", encoding = 'cp437')
    
    df = df.dropna()
    df = df.drop(['date','headlines','read_more'],1)
    df = df.set_index('author')
    df = df.reset_index(drop=True)

    print('extract configuration from input texts ...')
    Y = df.text
    X = df.ctext

    config = fit_text(X, Y)
    num_input_tokens = config['num_input_tokens']
    print('num is' + len(num_input_tokens))

    #summarizer = Seq2SeqSummarizer(config)

    


if __name__ == '__main__':
    main()
