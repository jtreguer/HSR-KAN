import torch
import numpy as np
import cv2
from cv2 import GaussianBlur
from einops import rearrange
import matplotlib.pyplot as plt #DEBUG


class NPZDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,get_graph=False):
        super(NPZDataset, self).__init__()
        loaded_data = np.load(data_path)
        self.LRHSI_list = loaded_data['LRHSI']
        self.RGB_list = loaded_data['RGB']
        self.GT_list = loaded_data['GT']

    def __getitem__(self, index):
        return torch.from_numpy(self.GT_list[index]).float(), \
                torch.from_numpy(self.LRHSI_list[index]).float(),\
                torch.from_numpy(self.RGB_list[index]).float()
    
    def __len__(self):
        return len(self.GT_list)
    
class ChikuseiDataset(torch.utils.data.Dataset):

    sigma = 0.5
    target_wavelengths = [0.49,0.56,0.665,0.89]
    sigma_filter = 0.05

    def __init__(self,full_image: np.array, training_zone: list,  wave_vector: np.array, device: torch.device, scale: int=4,gt_size: int=64):
        super().__init__()
        self.device = device
        self.full_image = full_image.astype(np.float32) # TYPE CONVERSION
        self.scale = scale
        self.gt_size = gt_size
        self.training_zone = training_zone #defined by (x0,y0,x1,y1)
        self.width = training_zone[2] - training_zone[0]
        self.height = training_zone[3] - training_zone[1]
        self.wave_vector = wave_vector  
        self.channels = full_image.shape[-1]
        self.subres = int(gt_size/scale)
        # In h w c order
        self.GT_list = self.make_gt()
        self.LRHSI_list = self.make_lr_hs()
        self.HRMSI_list = self.make_hr_ms()
        # Normalize GT
        for gt in self.GT_list:
            for k in range(len(self.wave_vector)):
                gt[:,:,k] = cv2.normalize(gt[:,:,k],None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
        self.dataset_size = len(self.GT_list)
        # Make tensor lists
        # In c h w order for the model to digest  
        self.GT_tensor_list = self.make_cuda_tensor(self.GT_list)
        self.LRHSI_tensor_list = self.make_cuda_tensor(self.LRHSI_list)
        self.HRMSI_tensor_list = self.make_cuda_tensor(self.HRMSI_list)
        # DEBUG
      #   lr = self.LRHSI_tensor_list[0].cpu().detach().numpy()[3,:,:]
      #   hr = self.HRMSI_tensor_list[0].cpu().detach().numpy()[3,:,:]
      #   print("DEBUG")
      #   plt.imshow(lr)
      #   plt.show()
      #   plt.imshow(hr)
      #   plt.show()


    def __getitem__(self, index):
          return self.GT_tensor_list[index], self.LRHSI_tensor_list[index], self.HRMSI_tensor_list[index]
    
    def __len__(self):
        return len(self.GT_list)
  
    def make_cuda_tensor(self, arr_list):
        tensor_list = []
        for arr in arr_list:
            arr = rearrange(arr,'h w c-> c h w')
            tensor_list.append(torch.from_numpy(arr).float().to(self.device))
        return tensor_list

    def make_lr_hs(self):
        # Gaussian blur, 3x3 kernel, then scale reduction
        lr_hs_chikusei = []
        for i in range(len(self.GT_list)):
            blurred_hs = GaussianBlur(self.GT_list[i],(3,3),sigmaX=self.sigma, borderType=0)
            if 1 :
                blurred_hs_normalized = np.empty(blurred_hs.shape, dtype=blurred_hs.dtype)
                for j in range(len(self.wave_vector)):
                    blurred_hs_normalized[:,:,j] = cv2.normalize(blurred_hs[:,:,j],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            if 0:
                 blurred_hs_normalized = cv2.normalize(blurred_hs[:,:,j],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            # Downsampling    
            lr_hs_chikusei.append(cv2.resize(blurred_hs_normalized,(self.subres,self.subres),interpolation=cv2.INTER_NEAREST))
        return lr_hs_chikusei
        
    def make_hr_ms(self):
        ms_list = []
        for j in range(len(self.GT_list)):
            ms = np.empty((self.gt_size, self.gt_size,len(self.target_wavelengths)))
            for i,wl in enumerate(self.target_wavelengths):
                filter = self.gaussian_response(self.wave_vector,wl,self.sigma_filter)
                filter /= np.max(filter)
                ms[:,:,i] = np.max(self.GT_list[j]*filter.reshape(1,1,len(filter)),axis=2)
                ms[:,:,i] = cv2.normalize(ms[:,:,i],None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            ms_list.append(ms)
        return ms_list

    def make_gt(self):
        i_range = self.height // self.gt_size
        j_range = self.width // self.gt_size
        # print(i_range, j_range)
        GT_list = []
        x0 = self.training_zone[0]
        y0 = self.training_zone[1]
        for i in range(i_range):
            for j in range(j_range):
                target_zone = self.full_image[y0+i*self.gt_size:y0+(i+1)*self.gt_size,x0+j*self.gt_size:x0+(j+1)*self.gt_size,:]
                GT_list.append(target_zone)
      #   print("First target zone")
      #   plt.matshow(GT_list[0][:,:,3]) 
        return GT_list
    
    def gaussian_response(self, x, mean, sigma):
        norm = 1/(sigma*np.sqrt(2*np.pi))
        return norm*np.exp(-0.5*((x-mean)/sigma)**2)


    


    

