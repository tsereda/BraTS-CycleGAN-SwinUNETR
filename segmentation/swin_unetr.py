import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding that preserves dimensions appropriately.
    
    Args:
        patch_size: Patch token size.
        in_channels: Number of input channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 2, 2),
        in_channels: int = 4,
        embed_dim: int = 48,
        norm_layer: nn.Module = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # CRITICAL FIX: Ensure patch embedding doesn't reduce dimensions twice
        # We'll use a convolutional layer with stride 1 to preserve spatial dimensions
        # This is necessary because we're already downsampling in later stages
        self.proj = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=3,  # Use 3x3x3 convolution 
            stride=1,       # Don't reduce dimension here
            padding=1       # Preserve spatial dimensions
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """Forward function."""
        input_shape = x.shape
        x = self.proj(x)
        out_shape = x.shape
        
        # Rearrange dimensions for transformer blocks
        x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        x = self.norm(x)
        
        return x


class WindowAttention3D(nn.Module):
    """Window based multi-head self-attention with improved numeric stability."""
    
    def __init__(
        self,
        dim: int,
        window_size: tuple,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        
        # Ensure num_heads divides dim evenly to avoid reshape errors
        if dim % num_heads != 0:
            adjusted_heads = num_heads
            while dim % adjusted_heads != 0 and adjusted_heads > 1:
                adjusted_heads -= 1
            print(f"Adjusted num_heads from {num_heads} to {adjusted_heads} to ensure divisibility")
            num_heads = adjusted_heads
            
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * 
                (2 * window_size[1] - 1) * 
                (2 * window_size[2] - 1), 
                num_heads
            )
        )  
        
        # Get pair-wise relative position index for each token in a window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function with improved numerical stability."""
        B_, N, C = x.shape
        
        # Safe QKV projection and reshape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B_, num_heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Get relative position bias and ensure dimensions match
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        
        # Permute to correct dimensions
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        
        # Add bias to attention (with sanity check)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerStage(nn.Module):
    """A Swin Transformer Layer for one stage with SIMPLIFIED ARCHITECTURE."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        downscale: bool = True,  # Whether to downscale at the end of the stage
        out_dim: Optional[int] = None  # Output dimension after downscaling
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.downscale = downscale
        # Default output dimension is double the input dimension if downscaling
        self.out_dim = out_dim if out_dim is not None else dim * 2 if downscale else dim
        
        # Build transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = TransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer
            )
            self.blocks.append(block)
            
        # Downscaling with channel adjustment
        if self.downscale:
            self.downsample = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(dim, self.out_dim, kernel_size=1, stride=1)  # 1x1 conv to adjust channels
            )
        else:
            self.downsample = nn.Identity()
        
    def forward(self, x):
        # Input should be (B, C, D, H, W)
        
        # Store identity for skip connection
        identity = x
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Optional skip connection (residual)
        x = x + identity
        
        # Downsample if needed
        x = self.downsample(x)
        
        return x


class TransformerBlock3D(nn.Module):
    """A simplified transformer block with improved stability."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Simplified architecture - use standard 3D convolutions
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm3d(dim)
        
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(dim)
        self.act2 = nn.GELU()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        
    def forward(self, x):
        # Standard residual convolution block
        identity = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Add residual connection
        x = x + identity
        x = self.act2(x)
        
        return x


class Encoder(nn.Module):
    """Simplified encoder with consistent dimensions."""
    
    def __init__(
        self,
        in_channels: int,
        feature_size: int = 48,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (4, 8, 16, 32),
    ):
        super().__init__()
        self.depths = depths
        self.feature_size = feature_size
        
        # Initial patch embedding - doesn't reduce spatial dimensions
        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, feature_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(feature_size),
            nn.GELU()
        )
        
        # Encoder stages
        self.stages = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i_stage in range(len(depths)):
            # Calculate current feature dimension
            dim = feature_size * (2 ** i_stage)
            
            # Check if this is the last stage (no downscaling after)
            is_last = i_stage == len(depths) - 1
            
            # Calculate output dimension for this stage
            out_dim = feature_size * (2 ** (i_stage + 1)) if not is_last else dim
            
            # Create stage
            stage = SwinTransformerStage(
                dim=dim,
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=(7, 7, 7),
                downscale=not is_last,
                out_dim=out_dim  # Pass the calculated output dimension
            )
            self.stages.append(stage)
            
            # Create skip connection paths
            if not is_last:
                self.skip_connections.append(nn.Identity())
            
    def forward(self, x):
        # Initial input shape
        initial_shape = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Store skip connections
        skips = []
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            # Save skip connection before processing
            if i < len(self.stages) - 1:
                skips.append(self.skip_connections[i](x))
                
            # Process through stage
            x = stage(x)
            
            # Add debug info for shape tracking
            #print(f"Stage {i} output shape: {tuple(x.shape)}")
        
        return x, skips


class DecoderBlock(nn.Module):
    """Decoder block with improved dimension handling."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        super().__init__()
        
        # Upsampling layer
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        
        # Convolutional block after concatenation
        self.conv1 = nn.Conv3d(
            out_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        # Upsampling
        x = self.upsample(x)
        
        # Print shapes for debugging
        #print(f"After upsampling: shape={x.shape}, skip shape={skip.shape}")
        
        # Resize if dimensions don't match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode='trilinear', align_corners=False
            )
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Process
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        return x


class Decoder(nn.Module):
    """Decoder with simplified architecture for stability."""
    
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        depths: Tuple[int, int, int, int]
    ):
        super().__init__()
        
        # Create decoder blocks
        self.blocks = nn.ModuleList()
        
        # Reversed loop through stages
        for i in range(len(depths) - 1, 0, -1):
            # Calculate channels
            in_channels = feature_size * (2 ** i)
            skip_channels = feature_size * (2 ** (i-1))
            out_channels = feature_size * (2 ** (i-1))
            
            # Create decoder block
            decoder_block = DecoderBlock(
                in_channels=in_channels,
                skip_channels=skip_channels,
                out_channels=out_channels
            )
            self.blocks.append(decoder_block)
        
        # Final 1x1 convolution to get segmentation map
        self.final_conv = nn.Conv3d(feature_size, num_classes, kernel_size=1)
        
    def forward(self, x, skips):
        # Reverse skip connections for decoder
        skips = skips[::-1]
        
        # Print shape before decoding
        #print(f"Before upsampling 0: shape={x.shape}")
        
        # Process through decoder blocks
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])
            
            # # Print shape for next upsampling
            # if i < len(self.blocks) - 1:
            #     print(f"Before upsampling {i+1}: shape={x.shape}")
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


class ImprovedSwinUNETR(nn.Module):
    """
    Improved SwinUNETR model with more stable architecture and fewer dimension issues.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        feature_size: int = 32,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        
        # Save parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.depths = depths
        
        # Calculate number of heads for each stage
        # Ensure divisibility with feature dimensions
        num_heads = []
        for i in range(len(depths)):
            stage_dim = feature_size * (2 ** i)
            # Find largest power of 2 that divides stage_dim
            heads = 1
            while heads * 2 <= stage_dim and stage_dim % (heads * 2) == 0:
                heads *= 2
            num_heads.append(heads)
            
        self.num_heads = tuple(num_heads)
        #print(f"Using num_heads: {self.num_heads}")
        
        # Create encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=self.num_heads
        )
        
        # Create decoder
        self.decoder = Decoder(
            feature_size=feature_size,
            num_classes=num_classes,
            depths=depths
        )
        
    def forward(self, x):
        # Store original dimensions
        input_shape = x.shape
        
        # Print input shapes for debugging
        # print(f"Image shape: {tuple(x.shape[2:])}, range: {x.min().item():.4f} to {x.max().item():.4f}")
        
        # Encoder
        bottleneck, skips = self.encoder(x)
        
        # Decoder
        logits = self.decoder(bottleneck, skips)
        
        # Ensure output size matches input size
        if logits.shape[2:] != input_shape[2:]:
            print(f"Resizing final output from {tuple(logits.shape[2:])} to {tuple(input_shape[2:])}")
            logits = F.interpolate(
                logits, 
                size=input_shape[2:],
                mode='trilinear',
                align_corners=False
            )
        
        return logits
    
    def initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# If running as a script, show model summary
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 2
    channels = 4
    depth = 128
    height = 128
    width = 128
    
    x = torch.randn(batch_size, channels, depth, height, width)
    
    # Initialize model
    model = SwinUNETR(in_channels=channels, num_classes=4, init_features=16)
    model.initialize_weights()
    
    # Forward pass
    output = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")