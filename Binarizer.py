import numpy as np

class Binarizer:
    def __init__(self):
        self.classes = {}
        self.word2Ind = {}
        self.Ind2word = {}
        
    def get_dict(self, data):
        idx = 0
        word2Ind = {}
        Ind2word = {}
        for sent in data:
            for word in sent:
                if word not in word2Ind.keys():
                    word2Ind[word] = idx
                    Ind2word[idx] = word
                    idx += 1
                    
        return list(word2Ind.keys()), word2Ind, Ind2word
        
    def fit(self, data):
        
        self.classes, self.word2Ind, self.Ind2word = self.get_dict(data)
    
    def transform(self,data):
        
        vector_lenght = max(self.Ind2word)+1
        y_matrix = np.zeros((len(data),vector_lenght))
        for row, sent in enumerate(data):
            for y_data in sent:
                if y_data in self.classes:
                    y_matrix[row, self.word2Ind[y_data]] = 1
        
        return y_matrix
    
    def get_vocab(self):
        
        return self.classes
        
        
    
