'''
Loading Data into script
'''
import torch
import numpy as np
import pandas as pd
import gensim.downloader
from tqdm import tqdm
from sklearn.utils import shuffle
from gensim.models import Word2Vec


class Dataset():

    '''
    dataset: the name of dataset to be loaded
    '''
    def __init__(self, dataset):
        self.vector = [] ### fixed-size vectors
        self.sentence = []
        self.label = []

        ##### -----------------------------------------Get Dataset---------------------------------
        df = pd.read_csv(dataset, names=['sentence', 'label'], encoding_errors = "ignore")
        ### Remove rows with missing value(s)
        df = df.dropna()
        
        # print(df['label'].value_counts())
        ##### -----------------------------------------Data Preprocessing--------------------------
        ### Replace the string label with numeric label
        df.label = pd.Categorical(df.label)
        df['label'] = df.label.cat.codes

        ### Drop minority class (too few samples)
        class_num = len(df['label'].unique())
        minority_class = []
        entries = len(df)
        for cls in range (class_num):
            if len(df.index[df['label'] == cls]) < entries/class_num:
                minority_class.append(cls)
        for mcls in minority_class:
            df = df.drop(df[df.label == mcls].index)

        ### Duplicate minority class
        df.label = pd.Categorical(df.label)
        df['label'] = df.label.cat.codes
        class_num = len(df['label'].unique())
        minority_class = []
        entries = len(df)
        for cls in range (class_num):
            if len(df.index[df['label'] == cls]) <= 70:
                minority_class.append(cls)

        for mcls in minority_class:
            df.append(df[df['label'] == mcls])
            # df = df.drop(df[df['label'] == mcls].index)
        df.label = pd.Categorical(df.label)
        df['label'] = df.label.cat.codes

        ### Shuffle the rows
        df = shuffle(df)

        ##### -----------------------------------------NLP------------------------------------------
        ### Split sentences into list by words
        corpus = [sent.split() for sent in df['sentence']]

        ### Get the max length of sentences
        self.max_len = 0
        for sent in corpus:
            if len(sent) > self.max_len:
                self.max_len = len(sent)

        vector_size = 5
        ### Convert Sentences into Matrix with Paddings (Making all matrix have the same size) (size = max_len to make the matrix a square)
        model = Word2Vec(corpus, min_count=1, vector_size= vector_size, window =4, sg = 1, epochs = 10)

        print("Preprocessing Sentences")
        for sent in tqdm(corpus):
            matrix = []
            for word in sent:
                matrix.append(model.wv[word])
            padding_flag = True
            if t := self.max_len//len(matrix)>1:
                for _ in range(t):
                    matrix += matrix
            for _ in range(self.max_len - len(matrix)): ### Padding
                if padding_flag:
                    matrix.insert(0, np.array([0 for _ in range(vector_size)], dtype='float32'))
                    padding_flag = not padding_flag
                else:
                    matrix.append(np.array([0 for _ in range(vector_size)], dtype='float32'))
                    padding_flag = not padding_flag

            self.vector.append(matrix)

        self.label = df.label.tolist()
        self.sentence = df.sentence.tolist()
        
    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        '''
        @param: index: int, location of data instance
        @return: sentence vector and label
        '''

        x = self.vector[index]
        y = self.label[index]

        return torch.tensor(x), torch.tensor(y), self.sentence[index]