import cv2
import glob
import numpy as np
from PIL import Image
from natsort import natsorted
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

pm = 6
path = "./results/samples/"
img_path = glob.glob(path + "/gamma/DoG/val/" + "*/*.png")
masked_path = glob.glob(path + "/gamma/DoG/masked/" + "*/*.png")
DoG_path = glob.glob(path + "/nogamma/DoG/edge/" + "*/*.png")
DoGinp_path = glob.glob(path + "/nogamma/DoG/inp/" + "*/*.png")
ganma_path = glob.glob(path + "/gamma/DoG/edge/" + "*/*.png")
gammainp_path = glob.glob(path + "/gamma/DoG/inp/" + "*/*.png")

def imsave(img, path):
    #im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im = Image.fromarray(img)
    im.save(path)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def SetImage(files):
    d = []
    for i in natsorted(files):
        img = Image.open(i)
        img = np.asarray(img)
        d.append(img)
    return d

def stitch_images(inputs, *outputs, img_per_row=1):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            #im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = np.array((images[cat][ix])).astype(np.uint8)
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img

print("set image")

img = SetImage(img_path)
masked = SetImage(masked_path)
DoG = SetImage(DoG_path)
DoGinp = SetImage(DoGinp_path)
gamma = SetImage(ganma_path)
gammainp = SetImage(gammainp_path)

results_path = "./results/samples"


for i in tqdm(range(166)):
    now = i*6
    nxt = (i+1)*6
    images = stitch_images(
        img[now:nxt],
        masked[now:nxt],
        DoG[now:nxt],
        DoGinp[now:nxt],
        gamma[now:nxt],
        gammainp[now:nxt],
        img_per_row = 1
    )

    path = os.path.join(results_path, "DoG")
    name = os.path.join(path, str(i).zfill(5) + ".png")
    create_dir(path)
    #print('\nsaving sample ' + name)
    images.save(name)
