"""
MiniMultiROCKET - From Scratch Implementation

A simplified implementation of ROCKET and MultiROCKET algorithms
for time series classification.

Classes:
    - miniminirocket: Uses only PPV feature
    - minimultirocket: Uses 4 features (PPV, MPV, MIPV, LSPV)
"""

from .miniminirocket import miniminirocket
from .minimultirocket import minimultirocket

__version__ = "0.1.0"
__author__ = "Noel Triguero"
__all__ = ["miniminirocket", "minimultirocket"]