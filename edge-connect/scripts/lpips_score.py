import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
from scipy.misc import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray
import lpips
import torchvision.transforms.functional as TF
import os

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--net', type=str, default="alex", help='Network for calculating lpips score')
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

if args.net == "alex":
    loss_fn = lpips.LPIPS(net="alex")
elif args.net == "vgg":
    loss_fn = lpips.LPIPS(net='vgg')

for d in range(0,7):
    files = list(glob(path_true + "/" + str(d) + '/*.jpg')) + list(glob(path_true + "/" + str(d) + '/*.png'))
    #print(files)
    alex = []
    names = []
    index = 1
    for fn in sorted(files):
        name = basename(str(fn))
        names.append(name)

        img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
        img_gt = (TF.to_tensor(img_gt) - 0.5) * 2
        img_gt.unsqueeze(0)
        img_pred = (imread(path_pred + "/" + str(d) + '/' + basename(str(fn))) / 255.0).astype(np.float32)
        img_pred = (TF.to_tensor(img_pred) - 0.5) * 2
        img_pred.unsqueeze(0)
        score = loss_fn(img_gt, img_pred)
        # print(score.item())
        # exit()
        alex.append(score.item())

        if np.mod(index, 100) == 0:
            print(
                str(index) + ' images processed',
                "LPIPS: %.4f" % round(np.mean(alex), 4),
            )
        index += 1

    print(
        "LPIPS: %.4f" % round(np.mean(alex), 4),
        "LPIPS Variance: %.4f" % round(np.var(alex), 4)
    )
