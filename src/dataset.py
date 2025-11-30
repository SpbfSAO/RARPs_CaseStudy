import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional


class FITS_Dataset(Dataset):
    """
    Class for loading and processing FITS image data.

    Args:
        data (np.ndarray): NumPy array containing image data.
                           Expected format (N, H, W, C) or (N, C, H, W).
        transform (Optional[Callable]): Optional transform function
                                       applied to each sample.
                                       Default is None.
    """
    def __init__(self, data: np.ndarray, transform: Optional[Callable] = None):
        """
        Initializes the dataset.

        Args:
            data (np.ndarray): Array of data.
            transform (Optional[Callable]): Data transformation.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")


        self.data = data
        self.transform = transform


    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)


    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a data sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Data sample as a PyTorch tensor.
        """
        sample = self.data[idx]


        if self.transform:
            sample = self.transform(sample)

        return torch.from_numpy(sample).float()
