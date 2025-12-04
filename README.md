# MiniMultiROCKET - From Scratch Implementation

A pure NumPy implementation of simplified miniROCKET (RandOm Convolutional KErnel Transform) algorithms for time series classification.

This repo is the result of a from-scratch implementation aimed at deeply understand the ROCEKT algorithm family and the Ridge Classifier. See `notebook.ipynb` for the complete development process.

## 📖 Overview

This project implements two variants of the ROCKET algorithm from scratch:

- **`miniminirocket`**: Uses only PPV (Proportion of Positive Values) feature
- **`minimultirocket`**: Uses 4 features (PPV, MPV, MIPV, LSPV) inspired by MultiROCKET

ROCKET is a state-of-the-art time series classification algorithm that transforms time series using random convolutional kernels and extracts statistical features from the resulting convolutions.

## ✨ Features

- 🚀 Pure NumPy implementation (no deep learning frameworks)
- 📊 Scikit-learn compatible API (`fit`, `transform`, `fit_transform`)
- 🎯 Multiple feature extraction strategies
- 💾 Lightweight and interpretable


## 🚀 Quick Start

```python
from minimultirocket import miniminirocket, minimultirocket
from sklearn.linear_model import RidgeClassifierCV

model1 = miniminirocket()
model2 = miniminirocket()
```

## 📊 Algorithm Details

### Kernel Generation
- **Kernel size**: 7
- **Kernel values**: α=-2, β=5
- **Combinations**: All possible positions for placing two β values

### Dilation Strategy
- Logarithmically spaced dilations
- Number of dilations: 16
- Adapts to input time series length

### Bias Calculation
- Computed from 5% sample of training data
- Uses 25th, 50th, and 75th percentiles

### Features Extracted

**miniminirocket** (1 feature per configuration):
- **PPV**: Proportion of values > bias

**minimultirocket** (4 features per configuration):
- **PPV**: Proportion of Positive Values
- **MPV**: Mean of Positive Values  
- **MIPV**: Mean of Indices of Positive Values
- **LSPV**: Longest Stretch of Positive Values


## 🛠️ Hyperparameter Tuning

The only hyperparameter is the Ridge classifier's alpha, we recommend using RidgeClassifierCV:

from sklearn.linear_model import RidgeClassifierCV

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))


## 📚 References

- Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery*, 34(5), 1454-1495.
https://arxiv.org/abs/1910.13051

- Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). MiniRocket: A very fast (almost) deterministic transform for time series classification. In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21)*, 248-257.
https://arxiv.org/abs/2012.08791

- Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiROCKET: multiple pooling operators and transformations for fast and effective time series classification. *Data Mining and Knowledge Discovery*, 36(5), 1623-1646.
https://arxiv.org/abs/2102.00457


## 👤 Author

**Noel Triguero Torres**
- GitHub: https://github.com/notrito
- LinkedIn: https://www.linkedin.com/in/noel-triguero-torres/

