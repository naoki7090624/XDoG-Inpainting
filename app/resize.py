import os
import cv2

path = "./imgs"
files = os.listdir(os.path.join(path,"original"))

for f in files:
    name = os.path.join(path,"original",f)
    img = cv2.imread(name)
    h = img.shape[0]
    w = img.shape[1]
    img = cv2.resize(img,(256,256))
    name = os.path.join(path,"resize",f)
    cv2.imwrite(name,img)