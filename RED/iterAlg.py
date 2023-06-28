import time
from tqdm.auto import trange
import numpy as np
from numpy import linalg as LA
import torch


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.detach().clone().cpu().numpy()

def to_tensor(x, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    if isinstance(x, np.ndarray):
        if len(x.shape) == 3:
            x = torch.from_numpy(x).unsqueeze(0).to(device)
        elif len(x.shape) == 4:
            x = torch.from_numpy(x).to(device)
        return x
    else:
        return x


def red(dObj, rObj, tau = 0.001, xinit=None, numIter=100, step=100, beta=1e-3, Lipz_total=1, backtracking=True, 
         backtotl=1, accelerate=False, xref=None, mask=None, clip=False):

    ##### HELPER FUNCTION #####
    evaluateTol = lambda x, xnext: np.linalg.norm(x.flatten('F') - xnext.flatten('F')) / np.linalg.norm(x.flatten('F'))
    evaluateGx = lambda s_step: 1 / Lipz_total * (dObj.grad(to_tensor(s_step), False) + tau * rObj(to_tensor(s_step), False))
    evaluatepsnr = lambda xtrue, x: 10 * np.log10(1 / np.mean((xtrue.flatten('F') - x.flatten('F'))**2))

    # log info
    timer = []
    relativeChange = []
    norm_Gs = []
    

    # initialize variables
    if xinit is None:
        with torch.no_grad():
            xinit = rObj.enhancer(dObj.y)

    x = xinit
    s = xinit
    t = 1.    # controls acceleration
    bar = trange(numIter)

    enhanced = [to_numpy(x)]

    if xref is not None:
        xrefSet = True
        rPSNR = [evaluatepsnr(xref, to_numpy(x))]
    else:
        xrefSet = False

    #Main Loop#
    for indIter in bar:
        timeStart = time.time()
        Gs = evaluateGx(to_tensor(s))
        xnext = to_numpy(s) - step * Gs
        xnext = np.clip(xnext, 0, np.inf) if clip else xnext  # clip to [0, inf]

        if mask is not None:
            xnext = mask * xnext

        norm_Gs.append(LA.norm(Gs.flatten('F')))

        timeEnd = time.time() - timeStart
        timer.append(timeEnd)

        if indIter == 0:
            relativeChange.append(np.inf)
        else:
            relativeChange.append(evaluateTol(to_numpy(x), xnext)) 

        # ----- backtracking (damping) ------ #
        if backtracking is True:
            G_update = evaluateGx(xnext)
            while LA.norm(G_update.flatten('F')) > LA.norm(Gs.flatten('F')) and step >= backtotl:
                step = beta * step
                # print(s.dtype, Gs.dtype)
                xnext = to_numpy(s) - step * Gs  
                xnext = np.clip(xnext, 0, np.inf) if clip else xnext
                G_update = evaluateGx(xnext)

                if step <= backtotl:
                    bar.close()
                    print("Reach to backtotl, stop updating.")
                    return to_numpy(x)

         # acceleration
        if accelerate:
            tnext = 0.5*(1+np.sqrt(1+4*t*t))
        else:
            tnext = 1

        s = xnext + ((t-1)/tnext)*(xnext-to_numpy(x))

        # update
        t = tnext
        x = xnext

        if xrefSet:
            rPSNR.append(evaluatepsnr(xref, to_numpy(x)))

        enhanced.append(to_numpy(x))

    bar.close()

    if xrefSet:
        return restore_best(rPSNR, enhanced)
    else:
        return to_numpy(x)

def restore_best(psnrs, enhanced):
    assert len(psnrs) == len(enhanced)
    return enhanced[np.argmax(psnrs)]