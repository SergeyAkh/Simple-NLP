import numpy as np
from scipy import sparse as sp_sparse

class BagOfWords:
    def __init__(self, maximum_words = None):
        self.dictionary = {}
        self.maximum_words = maximum_words
        
    def most_common_words(self, text):
        words_counts = {}

        for i in text:
            try:
                for j in i.split():
                    words_counts[j] = words_counts.get(j, 0) + 1
            except:
                for j in i:
                    words_counts[j] = words_counts.get(j, 0) + 1
                    
        dictionary = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
        
        dictionary = {k: v for k, v in dictionary}
        
        if self.maximum_words != None and len(dictionary) >= self.maximum_words:
            dictionary = dict(list(dictionary.items())[:self.maximum_words])
            # dictionary = {k: v for k, v in dictionary[:self.maximum_words]}

        dictionary = {v: k for k, v in enumerate(dictionary)}
        return dictionary
    
    def fit(self, text):
        
        self.dictionary = self.most_common_words(text)

    def transform(self, text):
        
        ALL_WORDS = self.dictionary.keys()
        TF_matrix = sp_sparse.lil_matrix((len(text),len(self.dictionary)))
                                
        for row,sent in enumerate(text):
            # if row % 100000 == 0:
            #     print(row)
            for word in sent.split():
                if word in ALL_WORDS:
                    
                    TF_matrix[row,self.dictionary[word]] = 1
        
        
        return TF_matrix
    
    def get_vocab(self):
        
        return self.dictionary
        
        
