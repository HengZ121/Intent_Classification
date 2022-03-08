'''
Loading Data into script
'''
import torch
import pandas as pd

from sklearn.utils import shuffle
from gensim.models import Word2Vec


class Dataset():

    '''
    dataset: the name of dataset to be loaded
    '''
    def __init__(self, dataset):
        self.sentence = [] ### fixed-size vectors
        self.label = []
        df = pd.read_csv(dataset, names=['sentence', 'label'], encoding_errors = "ignore")

        ### Remove rows with missing value(s)
        df = df.dropna()
        ### Shuffle the rows
        df = shuffle(df)

        print(df)

        ### Split sentences into list by words
        corpus = [sent.split() for sent in df['sentence']]
        
        # print(df['label'].value_counts())

        ### Replace the string label with numeric label
        df.label = pd.Categorical(df.label)
        df['label'] = df.label.cat.codes

        ### Get the max length of sentences
        self.max_len = 0
        for sent in corpus:
            if len(sent) > self.max_len:
                self.max_len = len(sent)

        ### Convert Sentences into Matrix with Paddings (Making all matrix have the same size) (size = max_len to make the matrix a square)
        model = Word2Vec(corpus, min_count=1, vector_size= self.max_len, window =3, sg = 1)
        
        self.sentence = []
        for sent in corpus:
            matrix = []
            for word in sent:
                matrix.append(model.wv[word])
            for _ in range(self.max_len - len(sent)): ### Padding
                matrix.append([0 for _ in range(self.max_len)])
            self.sentence.append(matrix)

        self.label = df.label.tolist()
        
    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        '''
        @param: index: int, location of data instance
        @return: sentence vector and label
        '''
        #Get quantity of children labels are desired in classification

        x = self.sentence[index]
        y = self.label[index]

        #print('-->',len(torch.tensor(input_ids[0:self.opt.sen_len])))
        return torch.tensor(x), torch.tensor(y)