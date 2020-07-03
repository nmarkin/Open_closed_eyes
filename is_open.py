import numpy as np
from pickle import load
from skimage.io import imread

pca = load(open('pca.pkl', 'rb'))
model = load(open('kmeans.pkl', 'rb'))


def is_open(path):
    image = imread(path)/255.0
    image = np.reshape(image, (1, -1))
    image = pca.transform(image)
    res = model.predict(image)
    if res in [0,1,2,4,6,9,14,15,17,20,21]:
        return 1
    else:
        return 0


print(is_open('EyesDataset/000009.jpg'))