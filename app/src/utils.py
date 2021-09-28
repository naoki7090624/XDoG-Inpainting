import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from flask import request
from PIL import Image
from io import BytesIO
import base64
from skimage.feature import canny
from skimage.color import rgb2gray
import numpy as np
import cv2


def DoG(img,size, sigma, k=1.6, gamma=1):
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
    return g1 - gamma*g2

def thres_dog(img, size, sigma, eps, k=1.6, gamma=0.98):
    d = DoG(img,size, sigma, k, gamma)
    d /= d.max()
    d *= 255
    d = np.where(d >= eps, 1, 0)
    return d

def XDoG(img, size, sigma, eps, phi, k=1.6, gamma=0.98):
    eps /= 255
    d = DoG(img, size, sigma, k, gamma)
    d /= d.max()
    e = 1 + np.tanh(phi*(d-eps))
    e[e>=1] = 1
    return e

def edge_detect(img, mask, model):
    mask = (1 - mask / 255).astype(np.bool)
    if model == "canny":
        #edge = canny(masked, sigma=2, mask=(1 - mask_gray).astype(np.bool)).astype(np.float)
        edge = canny(img, sigma=2, mask=mask).astype(np.float)
    elif model == "DoG":
        edge = thres_dog(img,7,1.5,30)
    else:
        edge = XDoG(img,9,1.3,30,10)
    
    return edge

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float().unsqueeze(0)
    return img_t

def preprocess_edge(images, edges, masks):
    edges_masked = edges * (1 - masks)
    images_masked = (images * (1 - masks)) + masks
    inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
    #print(inputs)
    return inputs

def preprocess_inp(images, edges, masks):
    images_masked = (images * (1 - masks).float()) + masks
    inputs = torch.cat((images_masked, edges), dim=1)
    return inputs


def postprocess(img):
    # img = img.permute(0,2,3,1) # [batch, height, width, ch]
    # img = img.to("cpu").detach().numpy().copy().squeeze()
    # img = Image.fromarray((img*255.0).astype(np.uint8))
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).int()
    img = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    return img

def DecodeImage(label):
    json = request.get_json()[label]
    dec = base64.b64decode(json.split(',')[1])
    img = Image.open(BytesIO(dec)).resize((256,256))
    return img

def EncodeImage(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    enc_img = base64.b64encode(buffer.getvalue()).decode("utf-8").replace("'", "")
    enc_img = "data:image/png;base64,{}".format(enc_img)
    return enc_img