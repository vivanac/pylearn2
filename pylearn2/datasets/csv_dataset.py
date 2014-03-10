# -*- coding: utf-8 -*-
"""
A simple general csv dataset wrapper for pylearn2.
Can do automatic one-hot encoding based on labels present in a file.
"""
__authors__ = "Zygmunt Zając"
__copyright__ = "Copyright 2013, Zygmunt Zając"
__credits__ = ["Zygmunt Zając"]
__license__ = "3-clause BSD"
__maintainer__ = "?"
__email__ = "zygmunt@fastml.com"

import csv
import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess


class CSVDataset(DenseDesignMatrix):
    """
    A generic class for accessing CSV files
    labels, if present, should be in the first column
    if there's no labels, set expect_labels to False
    if there's no header line in your file, set expect_headers to False
    """
    def __init__(self, 
            path = 'train.csv',
            preprocessor = None,
            one_hot = False,
            expect_labels = True,
            labels_for_regression = False,
            reverse_order_of_columns = False,            
            expect_headers = True,
            delimiter = ','):
        """
        .. todo::

            WRITEME
        """
        #one_hot=True
        #labels_for_regression = True

        self.path = path
        self.one_hot = one_hot
        self.expect_labels = expect_labels
        self.expect_headers = expect_headers
        self.delimiter = delimiter
        self.labels_for_regression = labels_for_regression
        self.reverse_order_of_columns = reverse_order_of_columns
        
        self.view_converter = None

        # and go

        self.path = preprocess(self.path)
        X, yt = self._load_data()        
        y = np.ndarray(shape=(yt.shape[0], 1), dtype='float32')
        y[:,0] = yt

        print 'X dim:{}'.format(X.ndim)
        print 'y dim:{}'.format(y.ndim)

        super(CSVDataset, self).__init__(X=X, y=y)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, False)

    def _load_data(self):
        """
        .. todo::

            WRITEME
        """
    
        assert self.path.endswith('.csv')
    
        if self.expect_headers:
            data = np.loadtxt(self.path, delimiter = self.delimiter, skiprows = 1)
        else:
            data = np.loadtxt(self.path, delimiter = self.delimiter)
        
        if self.expect_labels:

            if self.reverse_order_of_columns:
                num_col = data.shape[1]
                y = data[:,-1]
                X = data[:,0:num_col-1]
            else:
                y = data[:,0]
                X = data[:,1:] 
                        
            if self.one_hot:

                if self.labels_for_regression:
                    labels = y
                else:
                    # get unique labels and map them to one-hot positions
                    print '!!unique'
                    labels = np.unique(y)
                
                #labels = { x: i for i, x in enumerate(labels) }    # doesn't work in python 2.6
                labels = dict((x, i) for (i, x) in enumerate(labels))

                one_hot = np.zeros((y.shape[0], len(labels)), dtype='float32')
                print 'oen hot: y.shape[0]{}, labels.len:{}'.format(y.shape[0], len(labels))
                for i in xrange(y.shape[0]):
                    label = y[i]
                    label_position = labels[label]
                    one_hot[i,label_position] = 1.
                y = one_hot

        else:
            X = data
            y = None

        return X, y
