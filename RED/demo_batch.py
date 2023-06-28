import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_msssim import ms_ssim

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from ops import *
from iterAlg import red
from utils import *
from dataloading import RetinalDataset


def enhance_batch(model, y, xref=None, mask=None, tau=1e-5):
    ### Enhancing by the Generator
    print("enahancing by the network ...")
    with torch.no_grad():
        enhance_G = model(y)
    enhance_G_np = enhance_G.detach().clone().cpu().numpy()
    # del enhance_G
    # torch.cuda.empty_cache()
    print("done !")

    ### Enhancing by the PnP/RED
    print("enahancing by the network ...")
    loss_func = lambda x, y: 1.0 * (1 - ms_ssim(x, y, data_range=1.0, size_average=True))

    dObj = DataFidelityTorch(y, loss_func)
    rObj = Regularizer(model)

    enhanced_RED_np = red(dObj, rObj, 
                         step= 1 / (2 * tau + 1), numIter=300, 
                         backtracking=False, backtotl=1, 
                         accelerate=True, tau=tau, clip=False, 
                         Lipz_total=1.0, 
                         xref=xref.detach().numpy(), xinit=enhance_G, mask=mask)

    enhanced_RED_np = enhanced_RED_np
    enhanced_RED_np = np.clip(enhanced_RED_np, 0.0, 1.0) # clip to [0, 1]
    # enhanced_RED_np = max_norm(enhanced_RED_np)

    return enhance_G_np, enhanced_RED_np


if __name__ == '__main__':
    LC = True

    if LC:
        from model.model_LC import _NetG
    else:
        from model.model import _NetG

    mode = "low2high"
    device = torch.device("cuda:0")
    weight = join("./weight", "LC" if LC else "noLC" , mode, "exp10/model_denoise_199_45.pth")
    save_dir = join("./results", mode + "exp3")
    
    maybe_mkdir_p(save_dir)

    print("Intializing model ...")
    model = _NetG()
    model = nn.DataParallel(model).to(device)
    checkpoint = torch.load(weight)['model']
    model.load_state_dict(checkpoint.state_dict())
    model.eval()
    print("all states loaded ...")

    folders = {'deg': 'deg', 'pre':'pre', 'mask':'mask'}
    dataset = RetinalDataset("../degradation_test", folders)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False, pin_memory=False, num_workers=6)

    image_ids = []
    PSNRs = []
    SSIMs = []

    for idx, (degraded, ground_truth, mask, image_id) in enumerate(tqdm(dataloader)):
        bsz = degraded.size(0)
        image_ids += list(image_id)
        degraded_np = degraded.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
        degraded = degraded.to(device)
        enhance_G_np, enhanced_RED_np = enhance_batch(model, degraded, ground_truth, tau=1e-1)
        del degraded
        mask_np  = mask.detach().numpy()
        masked_enhance_G_np = (enhance_G_np * mask_np).transpose(0, 2, 3, 1)
        masked_enhanced_RED_np = (enhanced_RED_np * mask_np).transpose(0, 2, 3, 1)
        ground_truth_np = ground_truth.detach().numpy().transpose(0, 2, 3, 1)

        # print(masked_enhance_G_np.shape, masked_enhanced_RED_np.shape)
        # print(degraded_np.shape, ground_truth_np.shape)

        for i in range(bsz):
            degraded_np_case = degraded_np[i, ...]
            masked_enhance_G_np_case = masked_enhance_G_np[i, ...]
            masked_enhanced_RED_np_case = masked_enhanced_RED_np[i, ...]
            ground_truth_case = ground_truth_np[i, ...]

            PSNRs.append([
                        evaluatePSNR(degraded_np_case, ground_truth_case), 
                        evaluatePSNR(masked_enhance_G_np_case, ground_truth_case), 
                        evaluatePSNR(masked_enhanced_RED_np_case, ground_truth_case)
                    ])

            SSIMs.append([
                evaluateSSIM(degraded_np_case, ground_truth_case),
                evaluateSSIM(masked_enhance_G_np_case, ground_truth_case),
                evaluateSSIM(masked_enhanced_RED_np_case, ground_truth_case),
            ])

            ## save plot
            image_grid = get_image_grid([degraded_np_case.transpose(2, 0, 1), 
                                         masked_enhance_G_np_case.transpose(2, 0, 1), 
                                         masked_enhanced_RED_np_case.transpose(2, 0, 1), 
                                         ground_truth_case.transpose(2, 0, 1)], nrow=4).transpose(1, 2, 0)
            H, W, C = image_grid.shape
            
            image_grid = Image.fromarray((image_grid * 255.).astype(np.uint8))
            I1 = ImageDraw.Draw(image_grid)
            # Add Text to an image
            I1.text((W // 4 - 50, 2), f"PSNR={PSNRs[i][0]:.2f}dB\nSSIM={SSIMs[i][0]:.2f}", fill=(255, 255, 255))
            I1.text((2 * W // 4 - 50, 2), f"PSNR={PSNRs[i][1]:.2f}dB\nSSIM={SSIMs[i][1]:.2f}", fill=(255, 255, 255))
            I1.text((3 * W // 4 - 50, 2), f"PSNR={PSNRs[i][2]:.2f}dB\nSSIM={SSIMs[i][2]:.2f}", fill=(255, 255, 255))

            image_grid = image_grid.resize((W * 2, H * 2))
            maybe_mkdir_p(join(save_dir, "images"))
            image_grid.save(join(save_dir, "images", image_id[i] + ".png"), dpi=(500, 500))
            # Image.fromarray((degraded_np.transpose(1, 2, 0) * 255).astype(np.uint8)).resize((1024, 1024)).save("sample_poor.png", dpi=(500, 500))
            # Image.fromarray((enhanced_RED_np.transpose(1, 2, 0) * 255).astype(np.uint8)).resize((1024, 1024)).save("sample_enhanced.png", dpi=(500, 500))
    
    ## Save the results to a csv file 
    image_ids.append("avg")
    psnrs_mean, psnrs_std = np.mean(PSNRs, axis=0).tolist(), np.std(PSNRs, axis=0).tolist()
    ssims_mean, ssims_std = np.mean(SSIMs, axis=0).tolist(), np.std(SSIMs, axis=0).tolist()

    PSNRs.append([f"{psnrs_mean[i]:.4f}/{psnrs_std[i]:.4f}" for i in range(len(psnrs_mean)) ] )
    SSIMs.append([f"{ssims_mean[i]:.4f}/{ssims_std[i]:.4f}" for i in range(len(ssims_mean))] )
    
    df_res = np.concatenate([np.array(image_ids).reshape(-1, 1), np.array(PSNRs), np.array(SSIMs)], axis=-1)

    pd.DataFrame(df_res, columns=['Image ID', 'Degradation PSNR', 'Masked Enhanced PSRN', 'masked RED PSRN', 
                                    'Degradation SSIM', 'Masked Enhanced SSIM', 'Masked RED SSIM']).to_csv(join(save_dir, 'result.csv'), index=False)

