import os
from PIL import Image

for person in os.listdir(r"./Images_evaluation/"):
    person_path = os.path.join(r"./Images_evaluation/", person)
    os.makedirs(r"./Compressed_evaluation/" + person)
    for feet in os.listdir(person_path):
        image_path = os.path.join(person_path, feet)
        img = Image.open(image_path)
        img.save(r"./Compressed_evaluation/" + person + "/" + feet[0:-3] + "jpg")
        # print(feet[0:-3] + "jpg")
print("Success!")
