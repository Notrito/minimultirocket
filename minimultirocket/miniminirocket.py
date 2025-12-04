"""
miniminirocket.py

Implementation of MiniMiniROCKET (simplified miniROCKET) for time series classification.
This version uses only PPV (Proportion of Positive Values) as the feature.

Author: Your Name
"""

import numpy as np
import random
from itertools import combinations


class miniminirocket:
    """
    MiniMiniROCKET classifier using only PPV feature.
    
    A simplified version of ROCKET (RandOm Convolutional KErnel Transform) that uses:
    - Fixed kernel size: 7
    - Fixed kernel values: -2 and 5
    - Only PPV (Proportion of Positive Values) feature
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    kernel_size : int
        Size of the convolutional kernels (default: 7)
    num_dilations : int
        Number of dilation values to use (default: 16)
    kernels : list
        List of generated kernels
    dilations : np.ndarray
        Array of dilation values
    configurations : list
        List of (kernel, dilation, bias) tuples for feature extraction
    quantity_of_features : int
        Total number of features generated
    """
    
    def __init__(self):
        self.kernel_size = 7
        self.quantity_values = 2
        self.feature_value_1 = -2
        self.feature_value_2 = 5
        self.quantiles = [25, 50, 75]
        self.combinations = list(combinations(range(self.kernel_size), self.quantity_values))
        self.num_dilations = 16
        
    def calculate_ppv(self, convolution, bias):
        """
        Calculate Proportion of Positive Values.
        
        Parameters
        ----------
        convolution : list or np.ndarray
            Convolution result
        bias : float
            Threshold value
            
        Returns
        -------
        float
            Proportion of values in convolution that are greater than bias
        """
        biased = [1 if val > bias else 0 for val in convolution]
        ppv = sum(biased) / len(convolution)
        return ppv
    
    def calculate_biases_for_kernel_dilation(self, X, kernel, dilation):
        """
        Calculate bias values using quantiles from a sample of convolutions.
        
        Parameters
        ----------
        X : list of np.ndarray
            List of time series
        kernel : np.ndarray
            Convolutional kernel
        dilation : int
            Dilation value
            
        Returns
        -------
        list
            List of bias values (one per quantile)
        """
        n_samples = int(0.05 * len(X))
        indices = random.sample(range(len(X)), n_samples)
        all_sample_convolutions = []
        
        effective_length = (self.kernel_size - 1) * dilation + 1
        
        for idx in indices:
            conv = []
            for i in range(len(X[idx]) - effective_length + 1):
                window = X[idx][i : i + effective_length : dilation]
                res = window.dot(kernel)
                conv.append(res)
            all_sample_convolutions.extend(conv)
        
        biases = [np.percentile(all_sample_convolutions, q) for q in self.quantiles]
        return biases
    
    def calculate_all_convolutions(self, X, kernel, dilation):
        """
        Calculate convolutions for all time series with given kernel and dilation.
        
        Parameters
        ----------
        X : list of np.ndarray
            List of time series
        kernel : np.ndarray
            Convolutional kernel
        dilation : int
            Dilation value
            
        Returns
        -------
        list
            List of convolution results, one per time series
        """
        all_conv = []
        effective_length = (len(kernel) - 1) * dilation + 1
        
        for x in X:
            conv = []
            for i in range(len(x) - effective_length + 1):
                window = x[i : i + effective_length : dilation]
                res = window.dot(kernel)
                conv.append(res)
            all_conv.append(conv)
        
        return all_conv

    def fit(self, X):
        """
        Fit the transformer by generating kernels, dilations, and biases.
        
        Parameters
        ----------
        X : list of np.ndarray
            Training time series data
            
        Returns
        -------
        self
            Fitted transformer
        """
        l_input = len(X[0])
        max_exponent = np.log2(l_input - 1) / (self.kernel_size - 1)
        exponents = np.linspace(0, max_exponent, self.num_dilations)
        self.dilations = np.floor(2 ** exponents).astype(int)
        self.dilations = np.unique(self.dilations)

        # Generate kernels
        self.kernels = []
        for pos in self.combinations:
            kernel = np.zeros(self.kernel_size)
            kernel += self.feature_value_1
            kernel[pos[0]] = self.feature_value_2
            kernel[pos[1]] = self.feature_value_2
            self.kernels.append(kernel)

        # Store all configurations (kernel, dilation, bias)
        self.configurations = []
        
        for kernel in self.kernels:
            for dilation in self.dilations:
                biases = self.calculate_biases_for_kernel_dilation(X, kernel, dilation)
                for bias in biases:
                    self.configurations.append((kernel, dilation, bias))
        
        self.quantity_of_features = len(self.configurations)
        return self
        
    def transform(self, X):
        """
        Transform time series data into features.
        
        Parameters
        ----------
        X : list of np.ndarray
            Time series data to transform
            
        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        all_features = []

        for kernel, dilation, bias in self.configurations:
            all_convolutions = self.calculate_all_convolutions(X, kernel, dilation)
            
            ppvs = []
            for convolution in all_convolutions:
                ppv = self.calculate_ppv(convolution, bias)
                ppvs.append(ppv)
            
            all_features.append(ppvs)

        return np.array(all_features).T
    
    def fit_transform(self, X):
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : list of np.ndarray
            Training time series data
            
        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        return self.fit(X).transform(X)