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

    def __init__(self,full_image: np.array, training_zone: list, scale: int=4,gt_size: int=64):
        super().__init__()
        self.full_image = full_image
        self.scale = scale
        self.gt_size = gt_size
        self.training_zone = training_zone #defined by (x0,y0,x1,y1)
        self.width = training_zone[2] - training_zone[0]
        self.height = training_zone[3] - training_zone[1]
        self.GT_list = self.make_gt()
        self.LRHSI = self.make_lr_hs()
        self.HRMSI = self.make_hr_ms()
        

    def __getitem__(self, index):
        return torch.from_numpy(self.GT_list[index]).float(), \
                torch.from_numpy(self.LRHSI_list[index]).float(),\
                 torch.from_numpy(self.RGB_list[index]).float()
    
    def __len__(self):
        return len(self.GT_list)

    def make_lr_hs(self):
        # Gaussian blur, 3x3 kernel, then scale reduction
        subres = int(self.gt_size/self.scale)
        lr_hs_chikusei = []
        for i in range(len(self.GT_list)):
            blurred_hs = GaussianBlur(self.GT_list[i],(3,3),sigmaX=self.sigma, borderType=0)
            # Downsampling       
            lr_hs_chikusei.append(cv2.resize(blurred_hs,(subres,subres),interpolation=cv2.INTER_NEAREST))
        return lr_hs_chikusei
        

    def make_hr_ms(self):
        # TBC
        pass

    def make_gt(self):
        i_range = self.height // self.gt_size
        j_range = self.width // self.gt_size
        GT_list = []
        x0 = self.training_zone[0]
        y0 = self.training_zone[1]
        print(self.full_image.shape)
        print(i_range, j_range)
        for i in range(i_range):
            for j in range(j_range):
                GT_list.append(self.full_image[y0+i*self.gt_size:y0+(i+1)*self.gt_size,x0+j*self.gt_size:x0+(j+1)*self.gt_size,:])                
        return GT_list


    


    

