"""
minimultirocket.py

Implementation of MiniMultiROCKET for time series classification.
This version uses 4 features: PPV, MPV, MIPV, and LSPV.

Author: Your Name
"""

import numpy as np
import random
from itertools import combinations


class minimultirocket:
    """
    MiniMultiROCKET classifier using 4 features.
    
    A simplified version of MultiROCKET that uses:
    - Fixed kernel size: 7
    - Fixed kernel values: -2 and 5
    - 4 features: PPV, MPV, MIPV, LSPV
    
    Features:
    - PPV: Proportion of Positive Values
    - MPV: Mean of Positive Values
    - MIPV: Mean of Indices of Positive Values
    - LSPV: Longest Stretch of Positive Values
    
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
        Total number of features generated (4x base configurations)
    """
    
    def __init__(self):
        self.kernel_size = 7
        self.quantity_values = 2
        self.feature_value_1 = -2
        self.feature_value_2 = 5
        self.quantiles = [25, 50, 75]
        self.combinations = list(combinations(range(self.kernel_size), self.quantity_values))
        self.num_dilations = 16

    def calculate_features(self, convolution, bias):
        """
        Calculate all 4 features: PPV, MPV, MIPV, LSPV.
        
        Parameters
        ----------
        convolution : list or np.ndarray
            Convolution result
        bias : float
            Threshold value
            
        Returns
        -------
        tuple
            (ppv, mpv, mipv, lspv)
        """
        # 1. PPV - Proportion of Positive Values
        biased = [1 if val > bias else 0 for val in convolution]
        ppv = sum(biased) / len(convolution)
        
        # 2. MPV - Mean of Positive Values
        positive_values = [val for val in convolution if val > bias]
        if len(positive_values) == 0:
            mpv = 0
        else:
            mpv = sum(positive_values) / len(positive_values)

        # 3. MIPV - Mean of Indices of Positive Values
        positive_indices = [i for i, val in enumerate(convolution) if val > bias]
        if len(positive_indices) == 0:
            mipv = 0
        else:
            mipv = sum(positive_indices) / len(positive_indices)
        
        # 4. LSPV - Longest Stretch of Positive Values
        max_stretch = 0
        current_stretch = 0

        for val in convolution:
            if val > bias:
                current_stretch += 1
                max_stretch = max(max_stretch, current_stretch)
            else:
                current_stretch = 0
        lspv = max_stretch

        return ppv, mpv, mipv, lspv
    
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
        
        # Each configuration generates 4 features
        self.quantity_of_features = len(self.configurations) * 4
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
            
            # Separate lists for each feature
            ppvs = []
            mpvs = []
            mipvs = []
            lspvs = []
            
            for convolution in all_convolutions:
                ppv, mpv, mipv, lspv = self.calculate_features(convolution, bias)
                ppvs.append(ppv)
                mpvs.append(mpv)
                mipvs.append(mipv)
                lspvs.append(lspv)
            
            # Add each feature as a separate column
            all_features.append(ppvs)
            all_features.append(mpvs)
            all_features.append(mipvs)
            all_features.append(lspvs)

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