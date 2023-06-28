import numpy as np
from skimage.io import imread, imsave

from torch.utils.data import Dataset
from utils import join, subfiles


class RetinalDataset(Dataset):
    def __init__(self, dataroot, folders:dict):
        super().__init__()
        self.degrdation_folder = join(dataroot, folders['deg'])
        self.ground_truth_folder = join(dataroot, "pre")
        self.mask_folder = join(dataroot, "mask")

        self.degrdation_files = subfiles(self.degrdation_folder)
        

    def __len__(self):
        return len(self.degrdation_files)

    def __getitem__(self, idx):
        image_id = self.degrdation_files[idx].split("\\")[-1].split("_")
        subject_id = image_id[0] + "_" + image_id[1]
        image_id = (subject_id + "_" + image_id[2]).replace(".jpeg", "")

        degraded_np = np.float32(imread(self.degrdation_files[idx])).transpose(2, 0, 1) / 255.
        mask_np = np.float32(imread(join(self.mask_folder, subject_id + ".jpeg"))) / 255.
        ground_truth_np = np.float32(imread(join(self.ground_truth_folder, subject_id + ".jpeg"))).transpose(2, 0, 1) / 255.
        
        return degraded_np, ground_truth_np, mask_np[np.newaxis, ...], image_id

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    folders = {'deg': 'deg', 'pre':'pre', 'mask':'mask'}
    dataset = RetinalDataset("../degradation_test", folders)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for degraded_np, ground_truth_np, mask_np, image_id in dataloader:
        print(degraded_np.shape, ground_truth_np.shape, mask_np.shape, image_id)
        break