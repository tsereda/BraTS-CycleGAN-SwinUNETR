
"""
3D CycleGAN for Synthetic MRI Data Augmentation

Objective:
Generate synthetic missing modality data from 3D MRI volumes.
For preprocessed data (e.g., cropped to 128x128x128) with 3 channels.
Domain A: Complete volumes (all modalities present).
Domain B: Incomplete volumes generated on the fly by randomly dropping one modality.
The CycleGAN will learn to map from incomplete to complete and vice versa.

Environment Setup:
    conda create -n cyclegan3d_1 python=3.9
    conda activate cyclegan3d_1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install numpy nibabel matplotlib tqdm scipy
    pip install torchsummary
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchsummary import summary


# ------------------------------
# Environment and GPU Check
print("Checking environment...")
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    x = torch.randn(1).to(device)
    print("GPU is working. Device:", device)
except Exception as e:
    print("GPU test failed:", e)

# ------------------------------
# Dataset for loading preprocessed .npy volumes with optional modality dropout
class PreprocessedDataset(Dataset):
    def __init__(self, dir_path, simulate_dropout=False, dropout_mode="random", fixed_channel=0):
        """
        Args:
            dir_path (str): Directory with .npy files (each saved volume is shape (H,W,D,C)).
            simulate_dropout (bool): If True, randomly drop one channel.
            dropout_mode (str): "random" or "fixed". In "fixed", always drop the channel given by fixed_channel.
            fixed_channel (int): Which channel to drop in fixed mode.
        """
        self.files = sorted(glob(os.path.join(dir_path, '*.npy')))
        self.simulate_dropout = simulate_dropout
        self.dropout_mode = dropout_mode
        self.fixed_channel = fixed_channel

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        volume = np.load(file_path).astype(np.float32)  # May have shape (1,H,W,D,C) or (H,W,D,C)
        if volume.ndim == 5 and volume.shape[0] == 1:
            volume = volume[0]
        # Now volume is (H,W,D,C); transpose to (C,D,H,W)
        volume = np.transpose(volume, (3, 2, 0, 1))
        if self.simulate_dropout:
            C = volume.shape[0]
            if self.dropout_mode == "random":
                missing_channel = random.randint(0, C - 1)
            else:
                missing_channel = self.fixed_channel
            volume[missing_channel, ...] = 0.0
        return torch.from_numpy(volume)

# ------------------------------
# 3D Discriminator: PatchGAN
class Discriminator3D(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm3d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# ------------------------------
# 3D ResNet Block for Generator
class ResNetBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

# ------------------------------
# 3D Generator: Encoder-ResNet-Decoder
class Generator3D(nn.Module):
    def __init__(self, in_channels, out_channels, n_resnet=9):
        super(Generator3D, self).__init__()
        init_channels = 64
        model = [
            nn.Conv3d(in_channels, init_channels, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm3d(init_channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        curr_dim = init_channels
        for i in range(2):
            out_dim = curr_dim * 2
            model += [
                nn.Conv3d(curr_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(out_dim, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = out_dim
        for i in range(n_resnet):
            model += [ResNetBlock3D(curr_dim)]
        for i in range(2):
            out_dim = curr_dim // 2
            model += [
                nn.ConvTranspose3d(curr_dim, out_dim, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm3d(out_dim, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = out_dim
        model += [
            nn.Conv3d(curr_dim, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# ------------------------------
# Update image pool (avoid adding extra dimension)
def update_image_pool(pool, images, max_size=50):
    selected = []
    for image in images:
        # Check if image is batched (5D tensor: [B,C,D,H,W])
        if image.dim() == 4:
            image = image.unsqueeze(0)
        # Do not add an extra unsqueeze if already batched.
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random.random() < 0.5:
            selected.append(image)
        else:
            ix = random.randint(0, len(pool) - 1)
            selected.append(pool[ix])
            pool[ix] = image
    return torch.cat(selected, dim=0)

# ------------------------------
# Utility: Save generator models
def save_models(step, g_model_AtoB, g_model_BtoA, save_dir='./models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename1 = os.path.join(save_dir, 'g_model_AtoB_{:06d}.pth'.format(step+1))
    filename2 = os.path.join(save_dir, 'g_model_BtoA_{:06d}.pth'.format(step+1))
    torch.save(g_model_AtoB.state_dict(), filename1)
    torch.save(g_model_BtoA.state_dict(), filename2)
    print('>Saved models:', filename1, filename2)

# ------------------------------
# Utility: Visualize generated outputs (show middle slice)
def summarize_performance(step, g_model, dataset, name, n_samples=5, device='cpu'):
    g_model.eval()
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    for i, idx in enumerate(indices):
        vol = dataset[idx].unsqueeze(0).to(device)  # (1, C, D, H, W)
        with torch.no_grad():
            gen_vol = g_model(vol)
        vol_np = vol.cpu().numpy()[0]
        gen_np = gen_vol.cpu().numpy()[0]
        vol_np = (vol_np + 1) / 2.0
        gen_np = (gen_np + 1) / 2.0
        d_mid = vol_np.shape[1] // 2  # middle slice along depth axis
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(vol_np[0, d_mid, :, :], cmap='gray')
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(gen_np[0, d_mid, :, :], cmap='gray')
    # Fix: Use a writable directory instead of '/data/output/'
    import os
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, '{}_generated_plot_{:06d}.png'.format(name, step+1))
    plt.savefig(filename)
    plt.close()
    g_model.train()

# ------------------------------
# Training Loop for CycleGAN
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
          dataloader_A, dataloader_B, epochs=100, device='gpu'):
    
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    optimizer_G = optim.Adam(list(g_model_AtoB.parameters()) + list(g_model_BtoA.parameters()),
                             lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(d_model_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(d_model_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    pool_A = []
    pool_B = []
    
    lambda_id = 5
    lambda_cycle = 10
    
    g_model_AtoB.train()
    g_model_BtoA.train()
    d_model_A.train()
    d_model_B.train()
    
    n_steps = min(len(dataloader_A), len(dataloader_B))
    
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        progress_bar = tqdm(zip(dataloader_A, dataloader_B), total=n_steps)
        for i, (real_A, real_B) in enumerate(progress_bar):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            valid = torch.ones(real_A.size(0), 1, 1, 1, 1, device=device)
            fake = torch.zeros(real_A.size(0), 1, 1, 1, 1, device=device)
            
            # Train Generators
            optimizer_G.zero_grad()
            
            identity_A = g_model_BtoA(real_A)
            loss_id_A = criterion_identity(identity_A, real_A)
            identity_B = g_model_AtoB(real_B)
            loss_id_B = criterion_identity(identity_B, real_B)
            loss_identity = (loss_id_A + loss_id_B) * lambda_id
            
            fake_B = g_model_AtoB(real_A)
            loss_GAN_A2B = criterion_GAN(d_model_B(fake_B), valid)
            fake_A = g_model_BtoA(real_B)
            loss_GAN_B2A = criterion_GAN(d_model_A(fake_A), valid)
            loss_GAN = loss_GAN_A2B + loss_GAN_B2A
            
            recov_A = g_model_BtoA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = g_model_AtoB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) * lambda_cycle
            
            loss_G = loss_GAN + loss_cycle + loss_identity
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator A
            optimizer_D_A.zero_grad()
            loss_D_A_real = criterion_GAN(d_model_A(real_A), valid)
            fake_A_pool = update_image_pool(pool_A, [fake_A.detach()])
            loss_D_A_fake = criterion_GAN(d_model_A(fake_A_pool), fake)
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # Train Discriminator B
            optimizer_D_B.zero_grad()
            loss_D_B_real = criterion_GAN(d_model_B(real_B), valid)
            fake_B_pool = update_image_pool(pool_B, [fake_B.detach()])
            loss_D_B_fake = criterion_GAN(d_model_B(fake_B_pool), fake)
            loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
            loss_D_B.backward()
            optimizer_D_B.step()
            
            if (i+1) % 10 == 0:
                progress_bar.set_description("Iter {}/{}: D_A {:.3f}, D_B {:.3f}, G {:.3f}".format(
                    i+1, n_steps, loss_D_A.item(), loss_D_B.item(), loss_G.item()))
            
            if (i+1) % (n_steps // 1) == 0:
                summarize_performance(i, g_model_AtoB, dataloader_A.dataset, 'AtoB', device=device)
                summarize_performance(i, g_model_BtoA, dataloader_B.dataset, 'BtoA', device=device)
            if (i+1) % (n_steps * 5) == 0:
                save_models(i, g_model_AtoB, g_model_BtoA)
                
                
# ------------------------------
# Main Execution
if __name__ == '__main__':
    # Update these paths to match your system
    complete_data_dir = 'processed_data/brats128_cyclegan/images/'
    
    # Domain A: Complete data (simulate_dropout = False)
    dataset_A = PreprocessedDataset(complete_data_dir, simulate_dropout=False)
    # Domain B: Incomplete data (simulate_dropout = True, using random dropout)
    dataset_B = PreprocessedDataset(complete_data_dir, simulate_dropout=True, dropout_mode="random")
    
    dataloader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
    dataloader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)
    
    # Define input and output channels (here, 3 channels per volume)
    in_channels = 4
    out_channels = 4
    
    # Instantiate models.
    d_model_A = Discriminator3D(in_channels).to(device)
    d_model_B = Discriminator3D(in_channels).to(device)
    g_model_AtoB = Generator3D(in_channels, out_channels, n_resnet=6).to(device)
    g_model_BtoA = Generator3D(in_channels, out_channels, n_resnet=6).to(device)
    
    # Assuming your generator expects an input with shape (3, 128, 128, 128)
    summary(g_model_AtoB, input_size=(4, 128, 128, 128))

    # Begin training: for testing, we use 10 epochs. Adjust as needed.
    train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
          dataloader_A, dataloader_B, epochs=1, device=device)
