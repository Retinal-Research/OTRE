import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloading import RetinalDataset
from utils import *


if __name__ == '__main__':
    LC = True

    if LC:
        from model.model_LC import _NetG
    else:
        from model.model import _NetG

    mode = "low2high"
    device = torch.device("cuda:0")
    weight = join("./weight", "LC" if LC else "noLC" , mode, "exp10/model_denoise_199_45.pth")

    print("Intializing model ...")
    model = _NetG()
    model = nn.DataParallel(model).to(device)
    checkpoint = torch.load(weight)['model']
    model.load_state_dict(checkpoint.state_dict())
    model.eval()
    print("all states loaded ...")

    
    folders = {'deg': 'deg', 'pre':'pre', 'mask':'mask'}
    dataset = RetinalDataset("../degradation_test", folders)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False, pin_memory=False, num_workers=4)

    psnrs = []
    ssims = []
    with torch.no_grad():
        for idx, (degraded, ground_truth, mask, image_id) in enumerate(tqdm(dataloader)):
            # torch.cuda.empty_cache()
            enhanced = model(degraded.to(device))
            # del degraded, enhanced
        # ground_truth_np = ground_truth.clone().numpy().squeeze(0).transpose(1, 2, 0)
        # enhanced_np = enhanced.detach().clone().cpu().numpy().squeeze(0).transpose(1, 2, 0)
        # mask_np = mask.clone().numpy().squeeze(0).transpose(1, 2, 0)
        
        # psnrs.append(evaluatePSNR(enhanced_np, ground_truth_np))
        # ssims.append(evaluateSSIM(enhanced_np, ground_truth_np))

    # print(np.mean(psnrs), np.mean(ssims))