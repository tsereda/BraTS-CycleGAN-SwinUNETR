# Fix for the size mismatch in loss calculation

# Option 1: Resize the target tensor to match the discriminator output
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
            
            # Forward pass through discriminator once to get output shape
            with torch.no_grad():
                d_out_shape = d_model_A(real_A).shape
                
            # Create target tensors with proper size
            valid = torch.ones(d_out_shape, device=device)
            fake = torch.zeros(d_out_shape, device=device)
            
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

# Option 2: Alternative implementation - Modify the Discriminator to output a single value
class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_channels):
        super(PatchDiscriminator3D, self).__init__()
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

            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

class SingleValueDiscriminator3D(nn.Module):
    def __init__(self, in_channels):
        super(SingleValueDiscriminator3D, self).__init__()
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

            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1),
            
            # Add adaptive pooling to ensure a single output value
            nn.AdaptiveAvgPool3d(1)
        )
    
    def forward(self, x):
        return self.model(x)

# Usage with Option 2
# Replace d_model_A and d_model_B with:
# d_model_A = SingleValueDiscriminator3D(in_channels).to(device)
# d_model_B = SingleValueDiscriminator3D(in_channels).to(device)