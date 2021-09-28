# -*- encoding: utf-8 -*-
from flask import Flask, request, render_template, jsonify
import base64
from PIL import Image
from io import BytesIO
from src.networks import InpaintGenerator, EdgeGenerator
from src.utils import edge_detect, to_tensor, preprocess_edge, preprocess_inp, postprocess, EncodeImage, DecodeImage
import numpy as np
import torch
import os
from skimage.color import rgb2gray

app = Flask(__name__)

#NN作成
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

print(device)

@app.route('/')
def index():
    return render_template("content.html")


@app.route('/send_img', methods=['POST'])
def inpaiting():

    # Decode image and mask
    dec_img = DecodeImage("img")
    dec_mask = DecodeImage("mask")

    # Load Generators
    model = request.get_json()["model"] # canny, DoG, XDoG
    dataset = request.get_json()["dataset"] # celebA, celebA-HQ, paris
    path = os.path.join("./checkpoints", dataset, model)
    ckpt_Edge = os.path.join(path, "EdgeModel_gen.pth")
    ckpt_Inp = os.path.join(path, "InpaintingModel_gen.pth")

    if torch.cuda.is_available():
        data1 = torch.load(ckpt_Edge)
        data2 = torch.load(ckpt_Inp)
    else:
        data1 = torch.load(ckpt_Edge, map_location=lambda storage, loc: storage)
        data2 = torch.load(ckpt_Inp, map_location=lambda storage, loc: storage)

    EdgeNet = EdgeGenerator().to(device).eval()
    InpaintNet = InpaintGenerator().to(device).eval()
    EdgeNet.load_state_dict(data1['generator'])
    InpaintNet.load_state_dict(data2['generator'])

    # Prepare data
    img = np.array(dec_img.convert("RGB"))
    img_gray = rgb2gray(img) # 256*256, [0-1]
    mask = rgb2gray(np.array(dec_mask.convert("RGB")))
    mask = (mask > 0).astype(np.uint8) * 255
    edge = edge_detect(img_gray,mask,model)

    # numpy to tensor
    img = to_tensor(img).cuda()
    img_gray = to_tensor(img_gray).cuda()
    edge = to_tensor(edge).cuda()
    mask = to_tensor(mask).cuda()

    # Inpaint edge
    edge_input = preprocess_edge(img_gray, edge, mask)
    edge_output = EdgeNet(edge_input).detach()

    # Edge to Image transfer
    inpaint_input = preprocess_inp(img, edge_output, mask)
    inpaint_output = InpaintNet(inpaint_input)
    output = (inpaint_output * mask) + (img * (1-mask))

    # Post process output image
    output = postprocess(output)
    edge_output = postprocess(edge_output)

    # encode input image and output image
    enc_img = EncodeImage(dec_img)
    enc_edge = EncodeImage(edge_output)
    enc_result = EncodeImage(output)

    res = {
        'ip_type': 'inpaint_success',
        'img': enc_img,
        'edge': enc_edge,
        'result': enc_result
    }

    return jsonify(ResultSet=res)

if __name__ == '__main__':
    app.run(debug=True)
