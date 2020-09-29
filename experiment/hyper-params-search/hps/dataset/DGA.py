import pandas as pd
import numpy as np
import re
import random
from collections import Counter
import nltk
nltk.download('stopwords')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

## dnn check
from hps.ml.neural.DNN import DNN

class DGA(object):
    # general
    @classmethod
    def get_dataset(cls):
        # 기본 데이터 로드
        data = cls.load_data()
        #print(data)

        ## word2vec
        data['Domain Name'] = data['Domain Name'].map(lambda x : cls.cleanData(x))
        data = cls.delimiter(data)
        entire_corpus, true_corpus = cls.make_corpus_all_data(data)
        #entire_corpus = cls.shuffle_corpus(entire_corpus)
        true_corpus = cls.shuffle_corpus(true_corpus)

        # final data, embedded layer
        X_true_train, y_true_train, embedding_layer = cls.Embedding_words(data, true_corpus) # model.add (embedded_layer)
        #print(X_true_train, y_true_train, embedding_layer)

        return X_true_train, y_true_train, embedding_layer


    # data
    @staticmethod
    def load_data():
        data = pd.read_csv('DGA_dataset_labeling.csv')
        label_count = data["0(legit) / 1(dga)"].value_counts() # check
        #print(label_count) # 불균형
        return data


    # preprocessing
    ## stopwords 처리
    @staticmethod
    def cleanData(word):
        stopWords = stopwords.words('english')

        word = re.sub(r'[^A-Za-z0-9\s.]', r'', str(word).lower())
        word = re.sub(r'-', r'', word)

        word = " ".join([word for word in word.split() if word not in stopWords])
        return word


    # ascii to check no control characters
    ## ascii 변환 확인
    @staticmethod
    def to_ascii(data):
        ascii_ranges = ["-", ".", "0", "1", "9", "a", "z"]
        ascii_ord_ranges = list()
        for c in ascii_ranges :
            ascii_ord_ranges.append(ord(c))

        expected_ords = [ord("-"), ord(".")] + list(range(ord("0"), ord("9")+1)) + list(range(ord("a"), ord("z")+ 1))
        for i in ascii_ord_ranges:
            assert(i in expected_ords)

        valid_text_col = ['Domain Name']
        ord_data_dfs = dict() # ascii dict
        for col in valid_text_col :
            #print(col)
            ord_data_dfs[col] = data[col].apply(lambda  x: [ord(w) for w in x.lower()]).apply(pd.Series)
            #print(ord_data_dfs[col].head())



    # domain Name delimiter split
    @staticmethod
    def delimiter(data_df):
        split = data_df['Domain Name'].str.split('.')
        split = split.apply(lambda x: pd.Series(x))
        split_data = pd.concat([data_df, split], axis = 1)
        return split_data


    # tld
    ## 모든 split 행에서 마지막에 있는값이 tld
    @staticmethod
    def tld_list(data):
        tld_list = data['Domain Name'].str.split('.')
        temp = list()
        for i in range(len(tld_list)):
            temp.append(tld_list[i][-1])
        #print(Counter(temp))

        legit_tld_list = data[data['0(legit) / 1(dga)']==0]['Domain Name'].str.split('.')
        #print('legit_tld_list :', legit_tld_list)
        legit_temp = list()
        for i in range(len(legit_tld_list)):
            legit_temp.append(legit_tld_list[i][-1])
        #print(Counter(legit_temp))

    # corpus
    @staticmethod
    def make_corpus_all_data(data):
        ## generate / true값 저장
        tmp_corpus = data['Domain Name'].map(lambda x: x.split('.'))
        entire_corpus = []
        for i in range(len(tmp_corpus)):
            for line in tmp_corpus[i]:
                words = [x for x in line.split()]
                entire_corpus.append(words)

        ## true값만 저장
        tmp_true_corpus = list()
        for i in range(len(data)):
            if data['0(legit) / 1(dga)'][i] == 0 :
                tmp_true_corpus.append(data['Domain Name'][i].split('.'))

        return entire_corpus, tmp_true_corpus


    # shuffle corpus
    @staticmethod
    def shuffle_corpus(sentences):
        shuffled = list(sentences)
        random.shuffle(shuffled)
        return shuffled



    # word embedding
    @staticmethod
    def Embedding_words(data, word_list):
        t = Tokenizer()
        t.fit_on_texts(word_list)
        vocab_size = len(t.word_index) + 1

        X_encoded = t.texts_to_sequences(word_list)
        max_len = max(len(l) for l in X_encoded)

        ## embedding data
        X_train = pad_sequences(X_encoded, maxlen = max_len, padding = 'post')
        y_train = np.array(data['0(legit) / 1(dga)'])

        ## 임베딩 층 생성
        embedding_layer = Embedding(vocab_size, 19, input_size = 1048576)

        return X_train, y_train, embedding_layer

    # character embedding
    #def character_embedding(self, data):



if __name__ == "__main__":
    # dataset
    dga = DGA()
    ds_train, ds_test, embed_layer = dga.get_dataset()

    # model
    parameters = {
        "model_nm" : "DNN-test",
        "algorithm_type" : "classifier",
        "job_type" : "learn",
        ## learning parameters
        "global_step" : "10",
        "test_global_step" : "150",
        "early_type": "none",
        "min_step": "10",
        "early_key": "accuracy",
        "early_value": "0.98",
        ## algorithm parameters
        "input_units": "784",
        "output_units": "10",
        "hidden_units" : "100, 200, 100",
        "dropout_prob" : "0.1",
        "optimizer_fn" : "Adam",
        "learning_rate": "0.01",
        "initial_weight": "0.1",
        "act_fn" : "tanh"
    }
    dnn = DNN(parameters)
    dnn.build()

    dnn.learn(ds_train)
    print(dnn.predict(ds_test))











