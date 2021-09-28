import zipfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./datasets/test_mask.zip', help='path of zip file')
parser.add_argument('--dir', type=str, default='./test_mask', help='path of zip file')

args = parser.parse_args()
with zipfile.ZipFile(args.path,"r") as zip_ref:
    zip_ref.extractall(args.dir)
