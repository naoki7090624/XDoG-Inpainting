from scipy.misc import imread, imresize, imsave
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./datasets/paris_train_original', help='path of dataset')
parser.add_argument('--outpath', type=str, default='./datasets/paris/resize', help='save path')
parser.add_argument('--datasets', type=str, default="paris", help='dataset name (paris, celebA, mask)')

args = parser.parse_args()

files = os.listdir(args.path)
i = 0
for f in tqdm(files):
    name = os.path.join(args.path, f)
    im = imread(name)
    im = imresize(im, (256,256))
    if args.datasets == "paris":
        path = os.path.join(args.outpath,"{:05d}.png".format(i+1))
    elif args.datasets == "celebA":
        path = os.path.join(args.outpath,"{:06d}.jpg".format(i+1))
    else:
        path = os.path.join(args.outpath,"{:05d}.png".format(i))
    i+= 1
    imsave(path, im)
