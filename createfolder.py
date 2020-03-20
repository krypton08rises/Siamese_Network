import os
from os import path
from PIL import Image

for feet in os.listdir(r"./Images_evaluation/"):
    image_path = os.path.join(r"./Images_evaluation/", feet)
    img = Image.open(image_path)
    print(feet + feet[0:9])
    if not path.isdir(r"./Images_in_folder/" + feet[0:9]):
        os.makedirs(r"./Images_in_folder/" + feet[0:9])
    img.save(r"./Images_in_folder/" + feet[0:9] + "/" + feet)

for dir in os.listdir(r"./Images_in_folder"):
    count = 0
    for feet in os.listdir(os.path.join(r"./Images_in_folder", dir)):
        count+=1
    print(dir + ":" +str(count))
