"""ConPath: connectivity-calibrated stochastic occupancy research code.

The top-level package intentionally avoids importing PyTorch so that the exact NumPy label
oracle remains usable on machines where the training environment has not been created yet.
"""

__version__ = "0.2.0"
