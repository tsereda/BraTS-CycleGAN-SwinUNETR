import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.
    
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
        
        self.proj = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """Forward function."""
        x = self.proj(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.norm(x.permute(0, 2, 3, 4, 1))
        return x


class WindowAttention3D(nn.Module):
    """Window based multi-head self-attention module with relative position bias."""
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

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
        """Forward function.
        
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wd*Wh*Ww, Wd*Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self, 
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must be in 0 ~ window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward_part1(self, x, mask_matrix=None):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), "constant", 0)
        _, Dp, Hp, Wp, _ = x.shape
        
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
            
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B, D, H, W, C
        
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x
            
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]
            
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix=None):
        shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        if self.scale_by_keep:
            x = x.div(keep_prob) * random_tensor
        else:
            x = x * random_tensor
        return x


def window_partition(x, window_size):
    """Window partition function."""
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, window_size[0] * window_size[1] * window_size[2], C
    )
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """Window reverse function."""
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on feature size."""
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class SwinTransformerStage(nn.Module):
    """A basic Swin Transformer Layer for one stage."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: nn.Module = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = tuple(i // 2 for i in window_size)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # patch merging layer
        self.downsample = downsample
        
    def forward(self, x):
        # Calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        
        # calculate attention mask for SW-MSA
        Dp = int(np.ceil(D / self.window_size[0])) * self.window_size[0]
        Hp = int(np.ceil(H / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(W / self.window_size[2])) * self.window_size[2]
        
        attn_mask = None
        
        for blk in self.blocks:
            x = blk(x, attn_mask)
        
        x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x


class PatchMerging(nn.Module):
    """Patch merging layer."""

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)  # Changed from 4*dim to 8*dim
        self.norm = norm_layer(8 * dim)  # Changed from 4*dim to 8*dim

    def forward(self, x):
        """Forward function."""
        B, C, D, H, W = x.shape
        
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (D % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, W % 2, 0, H % 2, 0, D % 2))
        
        x = x.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B, D/2, H/2, W/2, C
        x1 = x[:, 0::2, 0::2, 1::2, :]  # B, D/2, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B, D/2, H/2, W/2, C
        x3 = x[:, 0::2, 1::2, 1::2, :]  # B, D/2, H/2, W/2, C
        x4 = x[:, 1::2, 0::2, 0::2, :]  # B, D/2, H/2, W/2, C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B, D/2, H/2, W/2, C
        x6 = x[:, 1::2, 1::2, 0::2, :]  # B, D/2, H/2, W/2, C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B, D/2, H/2, W/2, C
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B, D/2, H/2, W/2, 8*C
        x = self.norm(x)
        x = self.reduction(x)  # B, D/2, H/2, W/2, 2*C
        
        x = x.permute(0, 4, 1, 2, 3)  # B, 2*C, D/2, H/2, W/2
        
        return x


class Encoder(nn.Module):
    """Swin Transformer Encoder."""
    
    def __init__(
        self,
        in_channels: int,
        feature_size: int = 48,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        patch_size: Tuple[int, int, int] = (2, 2, 2),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.window_size = window_size
        self.feature_size = feature_size
        self.depths = depths
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=feature_size,
            norm_layer=nn.LayerNorm
        )
        
        # Swin Transformer stages
        self.stages = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        for i_stage in range(len(depths)):
            stage = SwinTransformerStage(
                dim=int(feature_size * 2 ** i_stage),
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging(dim=int(feature_size * 2 ** i_stage)) if i_stage < len(depths) - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(stage)
            
            # Create skip connections (adjust channels to match decoder)
            if i_stage < len(depths) - 1:  # Skip bottleneck
                out_channels = int(feature_size * 2 ** i_stage)
                self.skip_connections.append(
                    nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=1),
                        nn.GroupNorm(16, out_channels),
                        nn.PReLU(),
                    )
                )
            
    def forward(self, x):
        skip_outputs = []
        
        # Patch embedding
        x = self.patch_embed(x.contiguous())
        x = x.permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        
        # Swin Transformer stages
        for i_stage, stage in enumerate(self.stages):
            if i_stage < len(self.stages) - 1:  # Save skip connection
                skip_outputs.append(x)
            
            x = stage(x)
        
        # Process skip connections
        for i, skip in enumerate(skip_outputs):
            skip_outputs[i] = self.skip_connections[i](skip)
        
        return x, skip_outputs


class DecoderBlock(nn.Module):
    """Decoder block for SwinUNETR."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        use_batchnorm: bool = True
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        
    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x


class Decoder(nn.Module):
    """Decoder for SwinUNETR."""
    
    def __init__(
        self,
        feature_size: int = 48,
        num_classes: int = 4,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.depths = depths
        self.bottleneck_dim = int(feature_size * 2 ** (len(depths) - 1))
        
        # Up-sampling layers
        self.up_layers = nn.ModuleList()
        for i in range(len(depths) - 1, 0, -1):
            in_channels = int(feature_size * 2 ** i)
            out_channels = int(feature_size * 2 ** (i-1))
            self.up_layers.append(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(depths) - 1, 0, -1):
            in_channels = int(feature_size * 2 ** (i-1))
            skip_channels = int(feature_size * 2 ** (i-1))
            self.decoder_blocks.append(
                DecoderBlock(in_channels, in_channels, skip_channels)
            )
        
        # Final convolution
        self.final_conv = nn.Conv3d(feature_size, num_classes, kernel_size=1)
        
    def forward(self, x, skip_connections):
        # In reversed order for decoder
        skip_connections = skip_connections[::-1]
        
        for i, (up, decoder_block) in enumerate(zip(self.up_layers, self.decoder_blocks)):
            x = up(x)
            x = decoder_block(x, skip_connections[i])
        
        # Final convolution
        x = self.final_conv(x)
        
        return x  # Return the processed tensor

class SwinUNETR(nn.Module):
    """
    SwinUNETR model implementation.
    
    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        init_features: Initial feature size (feature_size in original implementation).
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        init_features: int = 16,  # Maps to feature_size
    ):
        super().__init__()
        
        # Save parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Default architecture parameters
        self.feature_size = init_features * 2  # Scale to match UNet3D capacity
        self.depths = (2, 2, 2, 2)
        self.num_heads = (3, 6, 12, 24)
        self.patch_size = (2, 2, 2)
        self.window_size = (7, 7, 7)
        
        # Create encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            feature_size=self.feature_size,
            depths=self.depths,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            window_size=self.window_size,
        )
        
        # Create decoder
        self.decoder = Decoder(
            feature_size=self.feature_size,
            num_classes=num_classes,
            depths=self.depths,
        )
        
    def forward(self, x):
        # Store original size for later upsampling
        original_size = x.shape[2:]
        
        # Encoder
        bottleneck, skip_connections = self.encoder(x)
        
        # Decoder
        logits = self.decoder(bottleneck, skip_connections)
        
        # Final upsampling to match target size
        if logits.shape[2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode='trilinear', align_corners=False)
        
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


# Print model summary if run directly
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