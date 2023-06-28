import os 
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class EyeQ_Dataset(Dataset):
    def __init__(self,mode,ori_path,deg_path,mask_path,transform_ori=None, transform_deg = None,transform_mask = None):
        
        all = ori_path + '*.*'
        self.image_list = glob.glob(all)

        self.original_root = ori_path
        self.degregation_root = deg_path
        self.mask_root = mask_path

        #self.image_list = glob.glob("dataset/segmentation/NewDrive/test/images/*.*")
        #self.original_root = 'dataset/segmentation/NewDrive/test/images/'
        #self.degregation_root = 'dataset/segmentation/NewDrive/test/deg/'

        # self.image_list = glob.glob("dataset/segmentation/NewIDRID/train/images/*.*")
        # self.original_root = 'dataset/segmentation/NewIDRID/train/images/'
        # self.degregation_root = 'dataset/segmentation/NewIDRID/train/deg/'
        
        self.transform_ori = transform_ori
        self.transform_deg = transform_deg
        self.transform_mask = transform_mask
        self.mode = mode

    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):

        original_path = self.image_list[idx]

        image_dir = os.path.split(original_path)[-1]
        image_or = os.path.splitext(image_dir)[0] +'.jpeg'
        image_root = os.path.splitext(image_dir)[0] + '_'+self.mode+'.jpeg'
        #print(image_name)

        degregation_path = self.degregation_root + image_root

        mask_path = self.mask_root + image_or


        
        original_image = Image.open(original_path)

        
        if self.transform_deg is not None:
            ori = self.transform_deg(original_image)


        deg_image = Image.open(degregation_path)


        if self.transform_ori is not None:

            deg = self.transform_ori(deg_image)


        mask_image = Image.open(mask_path)


        if self.transform_mask is not None:

            mask = self.transform_mask(mask_image)

        return ori,deg,mask,image_root