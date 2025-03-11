import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import time
from einops import rearrange
from tqdm import tqdm
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
# EXPERIMENTAL
from torch.amp import GradScaler, autocast

from utils import Metric,get_model_size,test_speed, set_logger,init_weights,set_seed, save_checkpoint, load_checkpoint
from models.KANFormer import KANFormer
from utils import get_model_size
from data import ChikuseiDataset

# Data
import rasterio
import spectral

# Specify the path to the ENVI data file and the file with .hdr
file = '/mnt/c/data/chikusei_ENVI/HyperspecVNIR_Chikusei_20140729.bsq'
header_file = '/mnt/c/data/chikusei_ENVI/HyperspecVNIR_Chikusei_20140729.hdr'

# Open the ENVI image using rasterio
with rasterio.open(file) as src:
    # Read the hyperspectral data into a NumPy array
    print("Reading HS data")
    hyperspectral_data = src.read()

    # Display information about the hyperspectral data
    print('Shape of hyperspectral data:', hyperspectral_data.shape)
    print('Number of bands:', src.count)

#Open the image with spectral
header_spectral = spectral.open_image(header_file)

# Access the wavelengths associated with each band
w_vector = np.array(header_spectral.bands.centers)

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_image = rearrange(hyperspectral_data,'c h w -> h w c')
chikusei_data = ChikuseiDataset(full_image=full_image,training_zone=[128,128,2176,1332],wave_vector=w_vector,device=device,scale=4,gt_size=64)

# Model
HSI_bands = full_image.shape[2]
MSI_bands = 4
chikusei_KAN = KANFormer(HSI_bands=HSI_bands,MSI_bands=MSI_bands,hidden_dim=256,scale=4,depth=4,image_size=64)
chikusei_KAN = chikusei_KAN.to(device)

# Training params
epochs = 1000
batch_size = 4
lr = 4e-4
loss_func = torch.nn.L1Loss()
optimizer = torch.optim.Adam(lr=lr,params=chikusei_KAN.parameters())
scheduler = StepLR(optimizer=optimizer,step_size=100,gamma=0.1) # Gamma set to 0.1 originally

full_image = rearrange(hyperspectral_data,'c h w -> h w c')
chikusei_data = ChikuseiDataset(full_image=full_image,training_zone=[128,128,1024,2048],wave_vector=w_vector,device=device,scale=4,gt_size=64)
train_dataloader = DataLoader(chikusei_data,batch_size=batch_size,drop_last=True,shuffle=True)

model_name = 'KANFormer'
scale = 4

# Single file save
def train(epochs: int,model: torch.nn.Module, checkpoint: str=None):

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled=False

    if checkpoint is None:
        print("No checkpoint - starting training from scratch")
        init_weights(model)
        log_dir = f'./trained_models/{model.name}_x{model.scale}/'
        print(log_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        filename = log_dir+f"{model_name}_x{model.scale}.pth"
        hist_batch_loss = []
        hist_epoch_loss = []

    if checkpoint is not None:
        filename = checkpoint
        print(f'Using check_point: {checkpoint}')
        cp = torch.load(checkpoint)
        model.load_state_dict(cp['model'],strict=False)  
        optimizer.load_state_dict(cp['optimizer']) 
        scheduler.load_state_dict(cp['scheduler'])
        hist_batch_loss = cp['hist_batch_loss']
        hist_epoch_loss = cp['hist_epoch_loss']

    model.train()

    training_start = time.time()
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        epoch_loss = 0

        # EXPERIMENTAL
        scaler = GradScaler('cuda')

        # Batch loop
        with tqdm(train_dataloader) as pbar:
            for idx,loader_data in enumerate(pbar):
                init_batch_loop = time.time()
                # t=  = time.time()
                GT,LRHSI,HRMSI = loader_data[0],loader_data[1],loader_data[2]
                # torch.cuda.synchronize()
                # print("Data loading time:", time.time()-t)
                # t = time.time()

                with autocast('cuda'):  # Use mixed precision EXPERIMENTAL
                    preHSI = chikusei_KAN(LRHSI,HRMSI)

                # torch.cuda.synchronize()
                # print("Forward pass time:", time.time()-t)
                # t = time.time()
                reg_loss =  chikusei_KAN.regularization_loss(regularize_activation=0.01, regularize_entropy=0.01)
                # torch.cuda.synchronize()
                # print("Reg loss computation time:", time.time()-t)
                # t = time.time()

                with autocast('cuda'): # Use mixed precision EXPERIMENTAL
                    loss = loss_func(GT,preHSI) + reg_loss

                # torch.cuda.synchronize()
                # print("Total loss computation time:", time.time()-t)
                optimizer.zero_grad(set_to_none=True) # None for faster allocation
                # t = time.time()

                # loss.backward()
                scaler.scale(loss).backward()

                torch.cuda.synchronize()
                # print("Backprop computation time:", time.time()-t)
                # t = time.time()

                # optimizer.step()

                scaler.step(optimizer)
                scaler.update()

                # torch.cuda.synchronize()
                # print("Optimizer step time:", time.time()-t)
                print(f"batch loop done in {time.time() - init_batch_loop}")
                hist_batch_loss.append(loss.item())
                pbar.set_postfix(epoch =epoch,loss=loss.item(), reg_loss=reg_loss, lr=optimizer.param_groups[0]['lr'])

        epoch_loss = np.mean(hist_batch_loss)
        print('Epoch loss:', epoch_loss)
        hist_epoch_loss.append(epoch_loss)
        # Save at the end of the epoch
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'hist_batch_loss': hist_batch_loss,
            'hist_epoch_loss': hist_epoch_loss
            }
    
        print(f"Saving: {filename}")   
        save_checkpoint(state, filename=filename)
        scheduler.step()
    print(f"Training with batch size {train_dataloader.batch_size} for {epochs} epochs completed in {time.time()-training_start:.2f}s")

    return hist_batch_loss, hist_epoch_loss

###############################################################
t = time.time()
torch.cuda.synchronize()
# b_loss, e_loss = train(100,chikusei_KAN,checkpoint='./trained_models/KANFormer_x4/KANFormer_x4.pth')
b_loss, e_loss = train(1200,chikusei_KAN,checkpoint=None)
print("Training time:", time.time()-t)
print(e_loss, len(e_loss))