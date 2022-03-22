'''
Loading Data into script
'''
from html import entities
import torch
import pandas as pd

from sklearn.utils import shuffle
from gensim.models import Word2Vec


class Dataset():

    '''
    dataset: the name of dataset to be loaded
    '''
    def __init__(self, dataset):
        self.sentence = [] 
        self.vector = [] ### fixed-size vectors
        self.label = []

        ##### -----------------------------------------Get Dataset---------------------------------
        df = pd.read_csv(dataset, names=['sentence', 'label'], encoding_errors = "ignore")
        ### Remove rows with missing value(s)
        df = df.dropna()
        
        # print(df['label'].value_counts())
        ##### -----------------------------------------Data Preprocessing--------------------------
        ### Identify minority class
        self.class_num = len(df['label'].unique())
        minority_class = []
        entries = len(df)
        for cls in range (self.class_num):
            if len(df.index[df['label'] == cls]) < entries/self.class_num:
                minority_class.append(cls)

        ### Drop minority class
        for mcls in minority_class:
            df = df.drop(df[df.label == mcls].index)

        ### Replace the string label with numeric label
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

        ### Convert Sentences into Matrix with Paddings (Making all matrix have the same size) (size = max_len to make the matrix a square)
        model = Word2Vec(corpus, min_count=1, vector_size= self.max_len, window =3, sg = 1)
        
        for sent in corpus:
            matrix = []
            for word in sent:
                matrix.append(model.wv[word])
            for _ in range(self.max_len - len(sent)): ### Padding
                matrix.append([0 for _ in range(self.max_len)])
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

        return torch.tensor(x), torch.tensor(y)