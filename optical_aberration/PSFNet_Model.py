import torch
import torch.nn as nn

class PSFNetCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for predicting Zernike coefficients from
    Point Spread Function (PSF) images.
    """
    def __init__(self, num_output_coeffs=3):
        """
        Initializes the layers of the network.

        Args:
            num_output_coeffs (int): The number of output Zernike coefficients
                                     to predict. Default is 3.
        """
        super(PSFNetCNN, self).__init__()

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Flatten layer to transition from conv to fully connected layers
        self.flatten = nn.Flatten()

        # Define the fully connected (linear) layers
        self.fc_layers = nn.Sequential(
            # Fully Connected Layer 1
            # The input size will depend on the image dimensions.
            # This will be calculated dynamically in the forward pass,
            # but we initialize it with a placeholder size first.
            nn.LazyLinear(out_features=128),
            nn.ReLU(),

            # Fully Connected Layer 2 (Output Layer)
            nn.Linear(in_features=128, out_features=num_output_coeffs)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor (batch of images).

        Returns:
            torch.Tensor: The model's output predictions.
        """
        # Pass input through convolutional layers
        x = self.conv_layers(x)

        # Flatten the output for the fully connected layers
        x = self.flatten(x)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        return x

if __name__ == '__main__':
    # Example of how to instantiate and test the model
    # This block will only run when the script is executed directly
    
    # Create a dummy input tensor to represent a batch of 4 images
    # Images are 64x64 pixels, with 1 channel (grayscale)
    dummy_input = torch.randn(4, 1, 64, 64) 

    # Instantiate the model
    model = PSFNetCNN(num_output_coeffs=3)
    
    # Pass the dummy input through the model
    output = model(dummy_input)
    
    # Print the model architecture and the output shape
    print("PSFNetCNN Model Architecture:")
    print(model)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    # Expected output shape: [4, 3] (batch_size, num_output_coeffs)
    assert output.shape == (4, 3)
    print("Model forward pass test successful!") 