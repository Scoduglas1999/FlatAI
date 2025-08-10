import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class PSFDataset(Dataset):
    """
    Custom PyTorch Dataset for loading PSF images and their corresponding
    Zernike coefficient labels.
    """
    def __init__(self, csv_file_path, images_dir_path, transform=None):
        """
        Args:
            csv_file_path (string): Path to the csv file with annotations.
            images_dir_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file_path)
        # To prevent KeyErrors from leading/trailing whitespace in column names,
        # we will programmatically strip whitespace from all column headers.
        self.labels_df.columns = self.labels_df.columns.str.strip()

        self.images_dir_path = images_dir_path
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (image, labels) where image is the PSF image tensor and
                   labels are the corresponding Zernike coefficients.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Retrieve image filename and labels from the DataFrame
        # Using .iloc for explicit integer-location based indexing which is more
        # robust with DataLoaders, especially in multiprocessing contexts.
        row = self.labels_df.iloc[idx]
        img_name = row['filename']
        labels = row[['z_4', 'z_7', 'z_8']]
        
        # 2. Construct the full path to the .npy image file
        img_path = os.path.join(self.images_dir_path, img_name)

        # 3. Load the image from the .npy file
        image = np.load(img_path)

        # 4. Convert NumPy image to a PyTorch tensor with a channel dimension
        # Assuming the input image is grayscale, shape [H, W] -> [1, H, W]
        # We explicitly convert the numpy array's dtype to float32 before creating
        # the tensor, as torch.from_numpy does not support the original uint16 type.
        image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        # 5. Normalize the image tensor's pixel values to the range [0, 1]
        # Assuming the images are 16-bit (max value 65535)
        image_tensor = image_tensor / 65535.0

        # 6. Create a PyTorch tensor of type float for the labels
        labels_tensor = torch.tensor(labels.values.astype(np.float32), dtype=torch.float)

        # 7. Return the tuple
        sample = (image_tensor, labels_tensor)

        if self.transform:
            sample = self.transform(sample)

        return sample 