import torch
import numpy as np
import cv2
from cv2 import GaussianBlur


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

    def __init__(self,full_image: np.array, training_zone: list,  wave_vector: np.array, scale: int=4,gt_size: int=64):
        super().__init__()
        self.full_image = full_image
        self.scale = scale
        self.gt_size = gt_size
        self.training_zone = training_zone #defined by (x0,y0,x1,y1)
        self.width = training_zone[2] - training_zone[0]
        self.height = training_zone[3] - training_zone[1]
        self.wave_vector = wave_vector  
        self.channels = full_image.shape[-1]
        self.subres = int(gt_size/scale)
        self.GT_list = self.make_gt()
        self.LRHSI_list = self.make_lr_hs()
        self.HRMSI_list = self.make_hr_ms()

    def __getitem__(self, index):

        return torch.from_numpy(self.GT_list[index].reshape((self.channels,self.gt_size,self.gt_size))).float(), \
                torch.from_numpy(self.LRHSI_list[index].reshape((self.channels,self.subres,self.subres))).float(),\
                torch.from_numpy(self.HRMSI_list[index].reshape((len(self.target_wavelengths),self.gt_size,self.gt_size))).float()
    
    def __len__(self):
        return len(self.GT_list)

    def make_lr_hs(self):
        # Gaussian blur, 3x3 kernel, then scale reduction
        lr_hs_chikusei = []
        for i in range(len(self.GT_list)):
            blurred_hs = GaussianBlur(self.GT_list[i],(3,3),sigmaX=self.sigma, borderType=0)
            # Downsampling       
            lr_hs_chikusei.append(cv2.resize(blurred_hs,(self.subres,self.subres),interpolation=cv2.INTER_NEAREST))
        return lr_hs_chikusei
        
    def normalize(self, image):
        curr_min = np.min(image)
        curr_max = np.max(image)
        r = curr_max - curr_min
        return (image-curr_min)/r

    def make_hr_ms(self):
        ms_list = []
        for j in range(len(self.GT_list)):
            ms = np.empty((self.gt_size, self.gt_size,len(self.target_wavelengths)))
            for i,wl in enumerate(self.target_wavelengths):
                filter = self.gaussian_response(self.wave_vector,wl,self.sigma_filter)
                filter /= np.max(filter)
                ms[:,:,i] = np.max(self.GT_list[j]*filter.reshape(1,1,len(filter)),axis=2)
                ms[:,:,i] = self.normalize(ms[:,:,i])
            ms_list.append(ms)
        return ms_list

    def make_gt(self):
        i_range = self.height // self.gt_size
        j_range = self.width // self.gt_size
        GT_list = []
        x0 = self.training_zone[0]
        y0 = self.training_zone[1]
        # print(self.full_image.shape)
        # print(i_range, j_range)
        for i in range(i_range):
            for j in range(j_range):
                GT_list.append(self.full_image[y0+i*self.gt_size:y0+(i+1)*self.gt_size,x0+j*self.gt_size:x0+(j+1)*self.gt_size,:])                
        return GT_list
    
    def gaussian_response(self, x, mean, sigma):
        norm = 1/(sigma*np.sqrt(2*np.pi))
        return norm*np.exp(-0.5*((x-mean)/sigma)**2)


    


    

