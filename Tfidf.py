import numpy as np
from scipy import sparse as sp_sparse

class TFIDF:
    def __init__(self):
        self.dictionary = {}

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
        
        return dictionary
    
    def fit(self, text):
        
        self.dictionary = self.most_common_words(text)
        
    def transform(self, text):
        
        every_words_list = [word[0] for word in self.dictionary]
        word_to_index = {word:index for index,word in enumerate(every_words_list)}
        length_rows = len(text)
        lenght_cols = len(every_words_list)
        TF_matrix = sp_sparse.lil_matrix((length_rows,lenght_cols))
        for row,sent in enumerate(text):
            # if row % 100000 == 0:
            #     print(row)
            for word in sent.split():
                if word in every_words_list:
                    TF_matrix[row,word_to_index.get(word,0)] += 1
        
        idf = np.log((length_rows+1)/(((TF_matrix>0)*1).sum(axis = 0) + 1))+1
        tfidf = TF_matrix.multiply(idf)
        norm_tfidf = tfidf.multiply(1/np.sqrt(np.sum(tfidf.power(2), axis = 1)).reshape((length_rows,1)))
        
        return sp_sparse.lil_matrix(norm_tfidf)
    
    def get_vocab(self):
        
        return self.dictionary
        
        