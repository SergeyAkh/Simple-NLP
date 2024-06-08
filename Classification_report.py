import numpy as np

class Report:
    def __init__(self):
        self.classes = {}
        self.word2Ind = {}
        self.Ind2word = {}
        
    def TP_FP(self, y_tr, y_pr):
        TP = np.sum(y_tr * y_pr)
        FP = np.sum((y_tr - y_pr) == -1)
        FN = np.sum((y_tr - y_pr) == 1)
        
        return TP, FP, FN
    
    def classification_report(self, y_true, y_pred):
        lenght = y_true.shape[1]
        TP = 0
        FP = 0
        FN = 0
        sum_of_wights = 0
        
        prec_comulative = 0
        sum_perc_class = 0
        
        rec_comulative = 0
        sum_rec_class = 0
        
        f1_comulative = 0
        sum_f1_class = 0
        
        for col in range(lenght):
            weight = np.sum(y_true[:,col])
            sum_of_wights += weight
            TP_col,FP_col,FN_col = self.TP_FP(y_true[:,col], y_pred[:,col])
        
            percision_per_class = TP_col/(TP_col + FP_col)
            sum_perc_class += percision_per_class * weight
            prec_comulative += percision_per_class
        
            recall_per_class = TP_col / (TP_col + FN_col)
            sum_rec_class += recall_per_class * weight
            rec_comulative += recall_per_class
        
            f1_per_class = 2*percision_per_class*recall_per_class / (percision_per_class + recall_per_class)
            sum_f1_class += f1_per_class * weight
            f1_comulative += f1_per_class
            
            TP += TP_col
            FP += FP_col
            FN += FN_col
            
        precision_macro = prec_comulative/lenght
        precision_micro = TP / (TP + FP)
        percision_weighted = sum_perc_class/sum_of_wights
        
        recall_macro = rec_comulative/lenght
        recall_micro = TP / (TP + FN)
        recall_weighted = sum_rec_class / sum_of_wights
        
        f1_macro = f1_comulative / lenght
        f1_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro)
        f1_weighted = sum_f1_class / sum_of_wights
        
        
        return  
        
        