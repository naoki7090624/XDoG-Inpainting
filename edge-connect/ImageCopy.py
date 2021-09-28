# Duplicate the image to see the result of different masks

import os
from tqdm import tqdm
import shutil
import random


def check_link(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)  # from out_dir to in_file
        shutil.copyfile(in_file, link_file)

data_path = "./datasets/paris/samples"
savedir = os.path.join(data_path, 'aug')
images_path = os.path.join(data_path, 'imgs')
files = os.listdir(images_path)

i = 0
for f in tqdm(files):
    #basename = "{:05d}.png".format(i+1)
    in_file = os.path.join(images_path, f)
    i+=1
    for j in range(13):
        basename = "{:05d}.png".format(i+j*72)
        link_file = os.path.join(savedir, basename)
        rel_link = os.path.relpath(in_file, savedir)
        shutil.copyfile(in_file, link_file)
        #check_link(images_path, basename, savedir)
