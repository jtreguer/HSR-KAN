
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from models.KAN import KANLinear


# KAN-CAB
class KANBlock(nn.Module):
    def __init__(self,input_dim,spline_order=3,grid_size=5):
        super(KANBlock, self).__init__()
        ### DEBUG
        #print("KANBlock initialized", spline_order,grid_size)
        self.kan_layer_01 = KANLinear(input_dim,input_dim,spline_order=spline_order,grid_size=grid_size)
        self.kan_layer_02 = KANLinear(input_dim,input_dim,spline_order=spline_order,grid_size=grid_size)

    def forward(self, x):
        shortcut = x
        # Global average pooling
        score = F.adaptive_avg_pool2d(x,(1,1))
        score = score.squeeze(-1).squeeze(-1)
        score = self.kan_layer_01(score)
        score = self.kan_layer_02(score)
        score = rearrange(score, 'b c -> b c 1 1')
        # Channel-wise multiplication
        x = x * score
        return x + shortcut

    
# KAN-Fusion
class Fusion(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,hidden_dim=256,scale=4,depth=4,image_size=64):
        super(Fusion, self).__init__()
        ### DEBUG
        #print("KANFUSION Initialized")
        self.hsi_kan = KANLinear(HSI_bands,hidden_dim)
        self.msi_kan = KANLinear(MSI_bands,hidden_dim)
        self.align_kan = KANLinear(hidden_dim*2,hidden_dim)
        self.scale = scale
        self.image_size = image_size

    def forward(self, LRHSI, HRMSI):
        # upscale LR HS
        print(LRHSI.shape)
        up_LRHSI = F.interpolate(LRHSI, scale_factor=self.scale, mode='bicubic', align_corners=True)
        print(up_LRHSI.shape)
        # reshape tensor from 31x16x16 to 256x31
        lrhsi_feats = rearrange(up_LRHSI, 'b c h w -> b (h w) c')
        # reshape tensor from 64x64x3 to 4096x3
        hrmsi_feats = rearrange(HRMSI, 'b c h w -> b (h w) c')
        print(lrhsi_feats.shape, hrmsi_feats.shape)
        print("lrhsi to first KAN")
        lrhsi_feats = self.hsi_kan(lrhsi_feats)
        print("hrmsi to second KAN")
        hrmsi_feats = self.msi_kan(hrmsi_feats)
        print(lrhsi_feats.shape, hrmsi_feats.shape)
        feats = torch.cat([lrhsi_feats, hrmsi_feats], dim=-1)  
        print("concatenated LR/HR to 3rd KAN")
        feats = self.align_kan(feats)  
        feats = rearrange(feats, 'b (h w) c -> b c h w', h=self.image_size)
        return feats
    
class KANFormer(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,hidden_dim=256,scale=4,depth=4,image_size=64):
        super(KANFormer, self).__init__()
        self.HSI_bands = HSI_bands
        self.MSI_bands = MSI_bands
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.fusion = Fusion(HSI_bands=HSI_bands,MSI_bands=MSI_bands,hidden_dim=hidden_dim,scale=scale,depth=depth,image_size=image_size)
        self.layers = nn.ModuleList([KANBlock(hidden_dim) for i in range(depth)])
        # Restructure
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, HSI_bands, 3,1,1)
        )

    def forward(self, LRHSI, HRMSI):
        up_HSI = F.interpolate(LRHSI, scale_factor=self.scale, mode='bicubic', align_corners=True)
        x = self.fusion(LRHSI, HRMSI)
        for layer in self.layers:
            x = layer(x)
        x = self.refine(x)
        return x + up_HSI

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum([layer.kan_layer_01.regularization_loss(regularize_activation, regularize_entropy)  + \
                    layer.kan_layer_02.regularization_loss(regularize_activation, regularize_entropy)    \
                    for layer in self.layers] )
    
if __name__== '__main__':

    model = KANFormer(HSI_bands=31,MSI_bands=3,hidden_dim=256,scale=2,depth=4,image_size=64)
