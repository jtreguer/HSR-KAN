import torch
import numpy as np


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

    def __init__(self,full_image: np.array, training_zone: list, scale: int=4):
        super().__init__()
        self.full_image = full_image
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
        pass

    def make_hr_ms(self):
        pass


    

