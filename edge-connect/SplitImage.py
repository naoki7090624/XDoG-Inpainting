import os
from tqdm import tqdm
import shutil
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./datasets/paris', help='dataset path')
parser.add_argument('--num_train', type=int, default=10000, help='how many train images')
parser.add_argument('--num_val', type=int, default=1000, help='how many validation images')
parser.add_argument('--num_test', type=int, default=1000, help='how many test images')
parser.add_argument('--datasets', type=str, default="paris", help='dataset name (paris, celebA, mask)')
args = parser.parse_args()

# check, if file exists, make link
def check_link(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)  # from out_dir to in_file
        shutil.copyfile(in_file, link_file)
        #os.symlink(in_file, link_file)
 
def add_splits(data_path):
    if args.datasets == "mask":
        subpath = "masks"
    else:
        subpath = "imgs"
    images_path = os.path.join(data_path, 'resize')
    train_dir = os.path.join(data_path, subpath, 'train')
    valid_dir = os.path.join(data_path, subpath, 'valid')
    test_dir = os.path.join(data_path, subpath, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
 
    # Split images into train, val, and test
    NUMS = args.num_train + args.num_val + args.num_test
    TRAIN_STOP = args.num_train
    VALID_STOP = args.num_train + args.num_val

    nums = list(range(0,NUMS))
    random.shuffle(nums)
    trains = nums[:TRAIN_STOP]
    vals = nums[TRAIN_STOP:VALID_STOP]
    tests = nums[VALID_STOP:NUMS]

    if args.datasets == "paris":
        for i in tqdm(trains):
            basename = "{:05d}.png".format(i+1)
            check_link(images_path, basename, train_dir)
        for i in tqdm(vals):
            basename = "{:05d}.png".format(i+1)
            check_link(images_path, basename, valid_dir)
        for i in tqdm(tests):
            basename = "{:05d}.png".format(i+1)
            check_link(images_path, basename, test_dir)
    elif args.datasets == "celebA":
        for i in tqdm(trains):
            basename = "{:06d}.jpg".format(i+1)
            check_link(images_path, basename, train_dir)
        for i in tqdm(vals):
            basename = "{:06d}.jpg".format(i+1)
            check_link(images_path, basename, valid_dir)
        for i in tqdm(tests):
            basename = "{:06d}.jpg".format(i+1)
            check_link(images_path, basename, test_dir)
    elif args.datasets == "mask":
        for i in tqdm(trains):
            basename = "{:05d}.png".format(i)
            check_link(images_path, basename, train_dir)
        for i in tqdm(vals):
            basename = "{:05d}.png".format(i)
            check_link(images_path, basename, valid_dir)
        for i in tqdm(tests):
            basename = "{:05d}.png".format(i)
            check_link(images_path, basename, test_dir)
    else:
        print("Cannot split this dataset. Please remake SplitImage.py") 
 
if __name__ == '__main__':
    add_splits(args.path)
