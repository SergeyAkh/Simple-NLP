import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_component = None):
        self.n_component = n_component
        self.eigenvalues = np.array(0)
        self.eigenvectors = np.array(0)
        
    def fit(self, data):
        cov_mat = np.cov(data.T)
        self.eigenvalues, self.eigenvectors = LA.eig(cov_mat)
        
    
    def pca_transform(self, data, comp = None):

        self.fit(data)

        if self.n_component is not None:
            if self.n_component <= data.shape[1]:
                two_vectors = self.eigenvectors[np.argsort(self.eigenvalues)[:self.n_component]]

            else:
                raise ValueError("Number of components > data dimention") 
        elif comp is not None:
            if comp <= data.shape[1]:
                two_vectors = self.eigenvectors[np.argsort(self.eigenvalues)[:comp]]

            else:
                raise ValueError("Number of components > data dimention") 
                
        else:
            self.n_component == data.shape[1]
            two_vectors = self.eigenvectors[np.argsort(self.eigenvalues)[:self.n_component]]
            
        return data@two_vectors.T
            
        
            
            
    def plot_explained_varience(self, data):
        
        # egnvalues, ei_vec = self.pca_manual(data)
    
        total_egnvalues = sum(self.eigenvalues)
        var_exp = [(i/total_egnvalues) for i in sorted(self.eigenvalues, reverse=True)]
        #
        # Plot the explained variance against cumulative explained variance
        #
        cum_sum_exp = np.cumsum(var_exp)
        plt.figure(dpi=100)
        plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5, align='center', 
                label='Individual explained variance',color='red')
        plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
    def plot_some_data(self, data, dict_words, names):
        num_of_rows = list(map(dict_words.get, names))
        data_to_plot = data[num_of_rows]
        A = [i[0] for i in data_to_plot]
        B = [i[1] for i in data_to_plot]
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Component_1")
        ax.set_ylabel("Component_2")
            
        ax.scatter(A, B)
        
        for i, txt in enumerate(names):
            ax.annotate(txt, (A[i], B[i]))
        plt.show()
        
    def get_param(self):
            
        return self.eigenvalues, self.eigenvectors
        
        
        
        
        