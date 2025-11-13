import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize



dir = './clf-data'
cat = ['empty', 'not_empty']

data = []
labels = []

for i, c in enumerate(cat):
    for file in os.listdir(os.path.join(dir, c)):
        img_path = os.path.join(dir, c, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(i)

data = np.array(data)
labels = np.array(labels)



