import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    Double convolution block for 3D UNet
    Improved to match Keras implementation more closely
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout3d(p=dropout_p)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, 
                 in_channels: int = 4, 
                 num_classes: int = 4, 
                 init_features: int = 16):
        """
        Improved 3D U-Net model to better match Keras implementation
        """
        super().__init__()
        
        # Save parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder path with Keras-matched dropout rates
        self.enc1 = ConvBlock3D(in_channels, init_features, dropout_p=0.1)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock3D(init_features, init_features * 2, dropout_p=0.1)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ConvBlock3D(init_features * 2, init_features * 4, dropout_p=0.2)
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = ConvBlock3D(init_features * 4, init_features * 8, dropout_p=0.2)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bridge
        self.bridge = ConvBlock3D(init_features * 8, init_features * 16, dropout_p=0.3)
        
        # Decoder path with Keras-matched dropout rates
        self.up1 = nn.ConvTranspose3d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(init_features * 16, init_features * 8, dropout_p=0.2)  # 16 = 8 (from up) + 8 (from enc4)
        
        self.up2 = nn.ConvTranspose3d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(init_features * 8, init_features * 4, dropout_p=0.2)   # 8 = 4 (from up) + 4 (from enc3)
        
        self.up3 = nn.ConvTranspose3d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(init_features * 4, init_features * 2, dropout_p=0.1)   # 4 = 2 (from up) + 2 (from enc2)
        
        self.up4 = nn.ConvTranspose3d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(init_features * 2, init_features, dropout_p=0.1)       # 2 = 1 (from up) + 1 (from enc1)
        
        # Final convolution
        self.final_conv = nn.Conv3d(init_features, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool4(enc4))
        
        # Decoder with skip connections
        # Up-conv, concatenate with skip connection, then convolution
        up1 = self.up1(bridge)
        dec1 = self.dec1(torch.cat([up1, enc4], dim=1))
        
        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, enc3], dim=1))
        
        up3 = self.up3(dec2)
        dec3 = self.dec3(torch.cat([up3, enc2], dim=1))
        
        up4 = self.up4(dec3)
        dec4 = self.dec4(torch.cat([up4, enc1], dim=1))
        
        # Final convolution
        logits = self.final_conv(dec4)
        return logits

    def initialize_weights(self):
        """Initialize model weights using He uniform initialization (like Keras)"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                # Use kaiming_uniform_ for He uniform initialization (matching keras he_uniform)
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# Print model summary if run directly
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 2
    channels = 3
    depth = 128
    height = 128
    width = 128
    
    x = torch.randn(batch_size, channels, depth, height, width)
    
    # Initialize model
    model = UNet3D(in_channels=channels, num_classes=4)
    model.initialize_weights()
    
    # Forward pass
    output = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")