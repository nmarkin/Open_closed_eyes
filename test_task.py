import os
import pickle
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

print(pickle.format_version)


images = os.listdir('EyesDataset/')
X, X_l = [], []
for name in images:
    image = imread('EyesDataset/' + name)
    # image_l = cv2.Laplacian(image, cv2.CV_64F)
    # X_l.append(image_l)
    X.append(image)

X = np.array(X)
# normalize data
X = X/255.0
X = X.reshape(4000, -1)

pca = PCA(n_components=28)
pca.fit(X)
res, i = 0, 0
for el in pca.explained_variance_ratio_:
    res += el
    i += 1
    print(i, el, res)

X_pca = pca.transform(X)

# pkl_filename = "pca.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(pca, file)

# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X_pca)
# X_tsne_df = pd.DataFrame(data=X_tsne, columns=['tsne-28d-one', 'tsne-28d-two'])

# df = pd.DataFrame(data=X_pca[:3], columns=['pca-one', 'pca-two', 'pca-three'])
#
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     palette=sns.color_palette("hls", 10),
#     data=df,
#     legend="full",
#     alpha=0.3
# )
# plt.show()
#
# ax = plt.figure().gca(projection='3d')
# ax.scatter(
#     xs=df["pca-one"],
#     ys=df["pca-two"],
#     zs=df["pca-three"],
#     cmap='tab10'
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')
# plt.show()


input_img = Input(shape=(24*24,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu', name='encoded')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(24*24, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X, X, epochs=1)

layer_name = 'encoded'
encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.get_layer(layer_name).output)
# X_encoded = encoder.predict(X)


model = KMeans(n_clusters=25, init='k-means++', random_state=241, n_jobs=-1)
model.fit(X_pca)

# pkl_filename = "kmeans.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)

# X_tsne_df['y'] = model.labels_
#
# sns.scatterplot(
#     x="tsne-28d-one", y="tsne-28d-two",
#     hue="y",
#     palette=sns.color_palette("hls", model.n_clusters),
#     data=X_tsne_df,
#     legend="full",
#     alpha=0.3
# )
# plt.show()

for i in range(model.n_clusters):
    fig = plt.figure(figsize=(8, 8))
    k = 0
    columns = 5
    rows = 5
    for j in range(len(X)):
        if model.labels_[j] == i:
            if k < columns*rows:
                img = X[j]
                img = img.reshape(24, 24)
                k += 1
                fig.add_subplot(rows, columns, k)
                plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    print(i)
    plt.show()
