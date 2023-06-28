import os
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


def enhance_case(model, y, xref=None, mask=None, tau=1e-3):
    ### Enhancing by the Generator
    print("enahancing by the network ...")
    enhance_G = model(y)
    enhance_G_np = enhance_G.detach().clone().cpu().numpy().squeeze(0)
    print("done !")

    ### Enhancing by the PnP/RED
    print("enahancing by the network ...")
    loss_func = lambda x, y: 1.0 * (1 - ms_ssim(x, y, data_range=1.0, size_average=True))
    # loss_func = lambda x, y: F.l1_loss(x, y)

    dObj = DataFidelityTorch(y, loss_func)
    rObj = Regularizer(model)

    enhanced_RED_np = red(dObj, rObj, step= 1 / (2 * tau + 1), numIter=600, backtracking=False, backtotl=1, accelerate=True, tau=tau, clip=False, Lipz_total=1.0, xref=xref.detach().numpy(), mask=mask)
    enhanced_RED_np = enhanced_RED_np.squeeze(0)
    enhanced_RED_np = np.clip(enhanced_RED_np, 0.0, 1.0) # clip to [0, 1]
    # enhanced_RED_np = max_norm(enhanced_RED_np)

    return enhance_G_np, enhanced_RED_np


if __name__ == '__main__':
    LC = True

    if LC:
        from model.model_LC import _NetG
    else:
        from model.model import _NetG,_NetD_256

    #mode = "low2high"
    mode = 'our_'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:0")
    #weight = "SottGan/Experiment/exp_full_5/checkpoint/model_denoise_200_60_20.pth"
    #weight = 'SottGan/Experiment/exp_full_5/checkpoint/model_denoise_77_60_20.pth'
    #weight = 'SottGan/Experiment/exp_full_5/checkpoint/model_denoise_85_60_20.pth'
    #weight = 'SottGan/Experiment/exp_full_4/checkpoint/model_denoise_17_60_20.pth'
    #weight = "SottGan/Experiment/exp_full_5/checkpoint/model_denoise_186_60_20.pth"
    weight = "SottGan/Experiment/exp_full_5/checkpoint/model_denoise_17_60_20.pth"
    #weight = 'SottGan/Experiment/exp11/checkpoint/model_denoise_65_45.pth'
    #weight = 'cycleGan/Experiment/exp_full/checkpoint/model_denoise_200_30.pth'
    #weight = 'cycleGan/Experiment/exp2/checkpoint/model_denoise_200_30.pth'
    #weight = 'OttGam/Experiment/exp1/checkpoint/model_denoise_200_50.pth'
    #weight = 'OttGam/Experiment/exp_full/checkpoint/model_denoise_200_30.pth'
    #weight = 'SottGan/Experiment/exp_full_5/checkpoint/model_denoise_165_60_20.pth'
    #weight = join("./weight", "LC" if LC else "noLC" , mode, "exp10/model_denoise_199_45.pth")
    save_dir = join("RED/experiment_IDRID", mode + "deg2high_8e-4")
    
    maybe_mkdir_p(save_dir)

    print("Intializing model ...")
    #model = _NetG()
    model = _NetG()
    model = nn.DataParallel(model).to(device)
    checkpoint = torch.load(weight)['model']
    model.load_state_dict(checkpoint.state_dict())
    model.eval()

    #model = torch.load(weight)["model"]
    print("all states loaded ...")

    #degradation_folder = join("../degradation_test", "deg")
    #degradation_folder = "dataset/degratation_test/deg"
    degradation_folder = "dataset/IDRID/deg"


    #ground_truth_folder = join("../degradation_test", "pre")
    #ground_truth_folder = "dataset/degratation_test/pre"
    #ground_truth_folder = "dataset/idrid_degraded/images"
    ground_truth_folder = "dataset/IDRID/images"

    #mask_folder = join("../degradation_test", "mask")
    #mask_folder = "dataset/degratation_test/mask"
    mask_folder = "dataset/IDRID/mask"


    degradation_files = subfiles(degradation_folder)

    image_ids = []
    PSNRs = []
    SSIMs = []

    for idx, file in enumerate(degradation_files):
        print(file)
        #image_id = file.split("\\")[-1].split("_")
        image_id = os.path.split(file)[-1].split("_")
        #os.path.split
        #name = image_id[1]
        #print(image_id)
        subject_id = image_id[0] + "_" + image_id[1]
        #print(image_id)
        image_id = (subject_id + "_" + image_id[2]).replace(".png", "")
        image_ids.append(image_id)
        # print(degradation_id)
        #print(file)
        mask_np = np.float32(imread(join(mask_folder, subject_id + ".png"))) / 255.
        
        degraded_np = np.float32(imread(file)).transpose(2, 0, 1) / 255.
        ground_truth_np = np.float32(imread(join(ground_truth_folder, subject_id + ".png"))).transpose(2, 0, 1) / 255.

        degraded_tensor = torch.from_numpy(degraded_np).unsqueeze(0).to(device)
        ground_truth_tensor = torch.from_numpy(ground_truth_np).unsqueeze(0)

        enhance_G_np, enhanced_RED_np = enhance_case(model, degraded_tensor, ground_truth_tensor, tau=8e-4)

        masked_enhance_G_np, masked_enhanced_RED_np = enhance_G_np * mask_np, enhanced_RED_np * mask_np

        PSNRs.append([
                        evaluatePSNR(degraded_np, ground_truth_np), 
                        evaluatePSNR(enhance_G_np, ground_truth_np), 
                        evaluatePSNR(masked_enhance_G_np, ground_truth_np), 
                        evaluatePSNR(enhanced_RED_np, ground_truth_np),
                        evaluatePSNR(masked_enhanced_RED_np, ground_truth_np)
                    ])

        SSIMs.append([
            evaluateSSIM(degraded_np.transpose(1, 2, 0), ground_truth_np.transpose(1, 2, 0)),
            evaluateSSIM(enhance_G_np.transpose(1, 2, 0), ground_truth_np.transpose(1, 2, 0)),
            evaluateSSIM(masked_enhance_G_np.transpose(1, 2, 0), ground_truth_np.transpose(1, 2, 0)),
            evaluateSSIM(enhanced_RED_np.transpose(1, 2, 0), ground_truth_np.transpose(1, 2, 0)),
            evaluateSSIM(masked_enhanced_RED_np.transpose(1, 2, 0), ground_truth_np.transpose(1, 2, 0)),
        ])

        ## save plot
        image_grid = get_image_grid([degraded_np, masked_enhance_G_np, masked_enhanced_RED_np, ground_truth_np], nrow=4).transpose(1, 2, 0)
        H, W, C = image_grid.shape
        
        image_grid = Image.fromarray((image_grid * 255.).astype(np.uint8))
        I1 = ImageDraw.Draw(image_grid)
        # Add Text to an image
        I1.text((W // 4 - 50, 2), f"PSNR={evaluatePSNR(degraded_np, ground_truth_np):.2f}dB\nSSIM={evaluateSSIM(degraded_np.transpose(1, 2, 0), ground_truth_np.transpose(1, 2, 0)):.2f}", fill=(255, 255, 255))
        I1.text((2 * W // 4 - 50, 2), f"PSNR={evaluatePSNR(masked_enhance_G_np, ground_truth_np):.2f}dB\nSSIM={evaluateSSIM(masked_enhance_G_np.transpose(1, 2, 0), ground_truth_np.transpose(1, 2, 0)):.2f}", fill=(255, 255, 255))
        I1.text((3 * W // 4 - 50, 2), f"PSNR={evaluatePSNR(masked_enhanced_RED_np, ground_truth_np):.2f}dB\nSSIM={evaluateSSIM(masked_enhanced_RED_np.transpose(1, 2, 0), ground_truth_np.transpose(1, 2, 0)):.2f}", fill=(255, 255, 255))

        image_grid = image_grid.resize((W * 2, H * 2))
        maybe_mkdir_p(join(save_dir, "images"))
        image_grid.save(join(save_dir, "images", image_id + ".png"), dpi=(500, 500))
        


        image_g0 = Image.fromarray((enhance_G_np.transpose(1, 2, 0) * 255.).astype(np.uint8))
        maybe_mkdir_p(join(save_dir, "GE_nomask"))
        image_g0.save(join(save_dir, "GE_nomask", subject_id + ".png"), dpi=(500, 500))

        image_g = Image.fromarray((masked_enhance_G_np.transpose(1, 2, 0) * 255.).astype(np.uint8))
        maybe_mkdir_p(join(save_dir, "GE"))
        image_g.save(join(save_dir, "GE", subject_id + ".png"), dpi=(500, 500))

        image_red0 = Image.fromarray((enhanced_RED_np.transpose(1, 2, 0)* 255.).astype(np.uint8))
        maybe_mkdir_p(join(save_dir, "REDE_nomask"))
        image_red0.save(join(save_dir, "REDE_nomask",  subject_id + ".png"), dpi=(500, 500))


        image_red = Image.fromarray((masked_enhanced_RED_np.transpose(1, 2, 0)* 255.).astype(np.uint8))
        maybe_mkdir_p(join(save_dir, "REDE"))
        image_red.save(join(save_dir, "REDE",  subject_id + ".png"), dpi=(500, 500))
        # Image.fromarray((degraded_np.transpose(1, 2, 0) * 255).astype(np.uint8)).resize((1024, 1024)).save("sample_poor.png", dpi=(500, 500))
        # Image.fromarray((enhanced_RED_np.transpose(1, 2, 0) * 255).astype(np.uint8)).resize((1024, 1024)).save("sample_enhanced.png", dpi=(500, 500))
        
        # if idx == 10:
        #     break
    
    ## Save the results to a csv file 
    # PSNRs, SSIMs = np.array(PSNRs), np.array(SSIMs)
    image_ids.append("avg")
    psnrs_mean, psnrs_std = np.mean(PSNRs, axis=0).tolist(), np.std(PSNRs, axis=0).tolist()
    ssims_mean, ssims_std = np.mean(SSIMs, axis=0).tolist(), np.std(SSIMs, axis=0).tolist()

    PSNRs.append([f"{psnrs_mean[i]:.4f}/{psnrs_std[i]:.4f}" for i in range(len(psnrs_mean)) ] )
    SSIMs.append([f"{ssims_mean[i]:.4f}/{ssims_std[i]:.4f}" for i in range(len(ssims_mean))] )
    
    df_res = np.concatenate([np.array(image_ids).reshape(-1, 1), np.array(PSNRs), np.array(SSIMs)], axis=-1)

    pd.DataFrame(df_res, columns=['Image ID', 'Degradation PSNR', 'Enhanced PSRN' , 'Masked Enhanced PSRN', 'RED PSRN', 'masked RED PSRN', 
                                    'Degradation SSIM', 'Enhanced SSIM', 'masked Enhanced SSIM', 'RED SSIM', 'masked RED SSIM']).to_csv(join(save_dir, 'result.csv'), index=False)

