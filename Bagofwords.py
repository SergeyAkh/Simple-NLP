import numpy as np
from scipy import sparse as sp_sparse

class BagOfWords:
    def __init__(self, maximum_words = None):
        """
        The parameters of logistic regression class.
        
        Argument:
        maximum_words -- the parameter that bounds with maximum of sorted by frequency dictionary 
        
        """
        self.dictionary = {}
        self.maximum_words = maximum_words
        
    def most_common_words(self, text):
        """
        This function calculates the frequency of appearance for each word
        
        Argument:
        text -- whole text corpus

        Returns:
        
        Dictionary of unique words and corresponding number
                
        """
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

        dictionary = {v: k for k, v in enumerate(dictionary)}
        return dictionary
    
    def fit(self, text): 
        """
        Function that runs process with data

        Argument:
        text -- whole text corpus

        Returns:

        Dictionary of unique words and corresponding number
        """
        
        self.dictionary = self.most_common_words(text)

    def transform(self, text):
        """
        Function that transform the data with created dictionary on step fit

        Argument:
        text -- whole text corpus (train or test)

        Returns:

        Matrix of the shape feeded data "text" with 1 on plases of words that is in the dictionary from step fit and 0 elsewhere
        """
        
        ALL_WORDS = self.dictionary.keys()
    
        TF_matrix = sp_sparse.lil_matrix((len(text),len(self.dictionary)))
                                
        for row,sent in enumerate(text):

            for word in sent.split():
                if word in ALL_WORDS:
                    
                    TF_matrix[row,self.dictionary[word]] = 1
        
        
        return TF_matrix
    
    def get_vocab(self):
        '''
        Returns created dictionary of words from step "fit"

        '''
        return self.dictionary
        
        
