import torch
import torch.nn as nn

class MultiScaleRepresentation(nn.Module):
    def __init__(self):
        super(MultiScaleRepresentation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=4)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)

    def forward(self, x):
        # Input x is a low-light RGB image I ∈ R^(H×W×C)
        # Applying the first convolution to generate Iq ∈ R^(H/4 × W/4 × 3)
        Iq = self.conv1(x)
        # Applying the second convolution on Iq to generate Io ∈ R^(H/8 × W/8 × 3)
        Io = self.conv2(Iq)
        
        return Iq, Io

# Usage
# Assuming input_image is your low-light RGB image tensor of shape (H, W, C)
# Reshape it to a tensor of shape (1, C, H, W) to fit into the Conv2d module
input_image = torch.randn(1, 3, 256, 256)  # Replace torch.randn with your actual input tensor

# Create an instance of MultiScaleRepresentation
model = MultiScaleRepresentation()

# Pass the input through the model
Iq_output, Io_output = model(input_image)
print(Iq_output.shape, Io_output.shape)
