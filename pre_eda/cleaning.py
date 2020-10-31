import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, scale

class Prep(object):
    '''
        Still In The Works!
    '''

    def __init__(self, data):
        self.data = data

    def remove_nan(self, how):
        if how == 'Rows':
            self.data.dropna()
            self.data = self.data.reset_index(drop = True)

        else:
            self.data.dropna(axis = 1)

    def replace_nan(self):
        pass

    def trans_pca(self):
        pass

    def trans_normal(self):
        pass

    def trans_scale(self):
        pass

    def trans_encode(self):
        pass

