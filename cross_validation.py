from sklearn.model_selection import cross_validate
# import numpy as np
# import pandas as pd


class TimeBasedCV(object):
    """ Time based cross-validator. Provides train/test indices to split time series data samples 
    that are observed at fixed time intervals, in train/test sets. 
    
    Args:
        train_period: number of time units to include in each train set (int, default = 4)
        test_period: number of time units to include in each test set (int, default = 1)
        date_column: name of column that represents time sequence (str)
    
    """
    
    def __init__(self, train_period=4, test_period=1, split_column='YEAR_MONTH'):
        self.train_period = train_period
        self.test_period = test_period
        self.split_column = split_column
     
        
    def split(self, data):
        """ Generate indices to split data into training and test set.
        
        Args:
            data: pandas DataFrame that contains one column as time sequence
        
        Returns:
            index_output: list of tuples (train index, test index) similar to sklearn model selection
        """         
        column_values = sorted(data[self.split_column].unique())
        train_indices_list = []
        test_indices_list = []
  
        train_start = 0
        train_stop = train_start + self.train_period
        test_start = train_stop
        test_stop = test_start + self.test_period
        
        while test_stop < len(column_values):
            cur_train_ind = data[data[self.split_column].isin(column_values[train_start  : train_stop])].index
            cur_test_ind = data[data[self.split_column].isin(column_values[test_start : test_stop])].index
            train_indices_list.append(cur_train_ind)
            test_indices_list.append(cur_test_ind)
            train_start += self.test_period
            train_stop  += self.test_period
            test_start += self.test_period
            test_stop  += self.test_period
            
#             print("Train period: {} - {}, Test period: {} - {}".format(train_start, train_stop, 
#                                                                        test_start, test_stop))
#             print("Train records: {}, Test records: {}".format(len(cur_train_ind), len(cur_test_ind)))

        # sklearn output  
        index_output = [(train,test) for train,test in zip(train_indices_list,test_indices_list)]

        self.n_splits = len(index_output)
        
        return index_output
    
    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator.
        Returns:
            n_splits : number of splitting iterations (int)
        """
        return self.n_splits 
    
    
def time_cross_val_scores(estimator, X, y, cv, scoring):
    """ Evaluate a score by time based cross-validation.
    
    Args:
        estimator: The object to use to fit the data
        X: The data to fit, array-like of shape (n_samples, n_features)
        y: The target variable to try to predict in the case of supervised learning. array-like of shape (n_samples,) 
        cv: Determines the cross-validation splitting strategy
        scoring: A str or a scorer callable object / function with signature scorer(estimator, X, y) 
        which should returns a score value.
    Returns:
        scores: Array of scores of the estimator for each run of the cross validation.
    """
    
    cv_results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        key : -cv_results[f'test_{key}'].mean()
        if key != 'R2' 
        else cv_results[f'test_{key}'].mean()
        for key in scoring.keys() 
    }