import torch, torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torchvision import transforms
from PIL import Image
import numpy as np
def tensorToImage(input_):
    input_ = (input_.clamp(-1, 1 ) + 1) / 2
    input_= (input_ * 255).type(torch.uint8)
    return(input_)

# define:
def encode(model, input_):
    model.eval()
    _, posterior = model(input_)
    mean = posterior.mode()
    return(mean)


def decode(model, input_):
    model.eval()
    recon = model.decode(input_)
    return(recon)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def crop_pad(fish_imgs, target_size = [203, 794]):
    fish_imgs= fish_imgs.astype(np.uint8)
    # Batch x Width x Heights x RGB
    sample, height, width, channels = fish_imgs.shape
    # height to pad
    pad_height = int(target_size[0]-height)
    # width to crop
    crop_width = int((width-target_size[1])/2)
    #print(fish_imgs[0:2][:,:,0:crop_width,:].shape, fish_imgs[0:2][:,:,-crop_width:,:].shape)
    # (439, 200, 78, 3) (439, 200, 78, 3)
    # new shape
    # first crop the image from width
    fish_imgs = fish_imgs[:, :, crop_width:-crop_width, :];
    # current shape
    current_shape = list(fish_imgs.shape)#
    current_shape[1] += pad_height
    # pad the image from height
    all_new_imgs= np.zeros(shape = tuple(current_shape), dtype = np.uint8)
    for i in range(sample):
        all_new_imgs[i, 0:height , :, :] = fish_imgs[i]
    return(all_new_imgs)

def obtain_matched_fac(qury_age_pl_mut, dummy_to_meta):
    mt_label = np.random.choice(qury_age_pl_mut.index)
    lb_count =  qury_age_pl_mut.loc[mt_label][0]
    age_pl_mut = mt_label.split('_') # label to image class, age, plate, mutant
    sample_class = [meta[meta['meta'].astype('str') == val]['dummy'].values
                            for meta,val in zip(dummy_to_meta, age_pl_mut)]
    return(sample_class, lb_count, mt_label)


def choose_loss(loss_opt = 'mse'):
    if loss_opt == 'mse':
        return(nn.MSELoss())
    elif loss_opt == 'huber':
        return(nn.HuberLoss())


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def update_config(config):
def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


##############################################################################
# Compute FID score
#############################################################################
from torchmetrics.image.fid import FrechetInceptionDistance
import scipy

def computeFID(imgs_true, imgs_pred, feature = 64, device = 'cuda'):
    """ function compute FID score between two set of images """
    torch.manual_seed(1234)
    fid = FrechetInceptionDistance(feature= feature).to(device)
    imgs_true = tensorToImage(imgs_true)
    imgs_pred = tensorToImage(imgs_pred)
    fid.update(imgs_true , real=True)
    fid.update(imgs_pred,real=False)
    score = fid.compute().cpu().numpy()
    return(score)


def sqrtm(input_data):
    m = input_data.detach().cpu().numpy().astype(np.float_)
    scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
    sqrtm_ = torch.from_numpy(scipy_res.real).to(input_data)
    return(sqrtm_)

def compute_fid(mu1, sigma1, mu2, sigma2, eps= 1e-6):
    """Adjusted version of `Fid Score`_

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        rank_zero_info(f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))
    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

# def compute_FID_emb(source_embs,target_embs):
#      """ function compute FID score between two set of embeddings(source, target), called compute_fid internelly and sqrtm"""
#     #source_mean = sampled_hidden;
#     #vae_hidden = sel_image_tr.to(config.device)
#     #moments for sampled hidden (source )
#     source_mean = torch.mean(source_embs, dim=0).view(-1)
#     source_embs = source_embs.view(source_embs.shape[0],-1)
#     source_cov = torch.cov(source_embs.t())
#     # target
#     target_mean = torch.mean(target_embs, dim=0).view(-1)
#     target_embs = target_embs.view(target_embs.shape[0],-1)
#     target_cov = torch.cov(target_embs.t())
#     fid = compute_fid(mu1 = source_mean,
#                                sigma1 = source_cov,
#                                mu2 = target_mean,
#                                sigma2 = target_cov
#                                )
#     return(fid)
