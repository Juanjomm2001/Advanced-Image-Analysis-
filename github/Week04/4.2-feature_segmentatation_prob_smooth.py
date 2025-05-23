#%%
# PREPARE
"""

Crear un programa que pueda identificar automáticamente diferentes regiones en una 
imagen basándose en su apariencia local (textura), no solo en la intensidad de los 
píxeles.




"""

#%% 


import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import local_features as lf
import scipy.ndimage

def ind2labels(ind):
    """ Helper function for transforming uint8 image into labeled image."""
    return np.unique(ind, return_inverse=True)[1].reshape(ind.shape)

path = '../Week04/week4/3labels/' # Change path to your directory

# READ IN IMAGES
training_image = skimage.io.imread(path + 'training_image.png')
training_image = training_image.astype(float)
training_labels = skimage.io.imread(path + 'training_labels.png')

# 4.2.2 (B) Prepare labels for clustering
training_labels = ind2labels(training_labels) #te pasa a etiqueta cada pixel
print(training_labels)
nr_labels = np.max(training_labels)+1 # number of labels in the training image
print(nr_labels)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(training_image, cmap=plt.cm.gray)
ax[0].set_title('training image')
ax[1].imshow(training_labels)
ax[1].set_title('labels for training image')
fig.tight_layout()
plt.show()






#%% 
# TRAIN THE MODEL

#DICCIONARIO VISUAL 

#este sigma hace que mire los vecinos al rededor para entender la textura en al que s eencuentra el pixel
sigma = [1, 2 , 3] #clases para calcular caracterisitcas gausianas
features = lf.get_gauss_feat_multi(training_image, sigma)
features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))
labels = training_labels.ravel()

#selecta arandom subset
nr_keep = 15000 # number of features randomly picked for clustering 
keep_indices = np.random.permutation(np.arange(features.shape[0]))[:nr_keep]

features_subset = features[keep_indices]
labels_subset = labels[keep_indices]



# Agrupa los 15,000 píxeles en 1,000 grupos o clusters  según sus "huellas digitales"
# Cada grupo contiene píxeles que "se parecen" en términos de textura
nr_clusters = 1000 # number of feature clusters
# for speed, I use mini-batches
kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=nr_clusters, 
                                         batch_size=2*nr_clusters, 
                                         n_init='auto')
kmeans.fit(features_subset)
assignment = kmeans.labels_


# Para cada clúster, calcula qué fracción de sus miembros pertenece a cada etiqueta

edges = np.arange(nr_clusters + 1) - 0.5 # histogram edges halfway between integers
hist = np.zeros((nr_clusters, nr_labels))

# Para cada grupo de píxeles similares, cuenta cuántos pertenecen a cada clase
# Ejemplo: Grupo 1 tiene 100 píxeles:

# 80 son de "hueso"
# 15 son de "cartílago"
# 5 son de "fondo"

# Entonces el Grupo 1 tiene:

# 80% probabilidad de ser "hueso"
# 15% probabilidad de ser "cartílago"
# 5% probabilidad de ser "fondo"
for l in range(nr_labels):
    hist[:, l] = np.histogram(assignment[labels_subset == l], bins=edges)[0]
sum_hist = np.sum(hist, axis=1)
cluster_probabilities = hist/(sum_hist.reshape(-1, 1))

fig, ax = plt.subplots(2, 1)
legend_label = [f'label {x}' for x in range(nr_labels)]

ax[0].plot(hist, '.', alpha=0.5, markersize=3)
ax[0].set_xlabel('cluster id')
ax[0].set_ylabel('number of features in cluster')
ax[0].legend(legend_label)
ax[0].set_title('features in clusters per label')
ax[1].plot(cluster_probabilities, '.', alpha=0.5, markersize=3)
ax[1].set_xlabel('cluster id')
ax[1].set_ylabel('label probability for cluster')
ax[1].legend(legend_label)
ax[1].set_title('cluster probabilities')
fig.tight_layout()
plt.show()


# Ahora tienes un diccionario que dice: "Si veo un píxel con esta textura 
# (Grupo X), hay Y% de probabilidad de que sea de la clase Z"

# Finished training

#%% 
# USE THE MODEL paso D

# 2. Extraer las "huellas digitales" de la nueva imagen
testing_image = skimage.io.imread(path + 'testing_image.png')
testing_image = testing_image.astype(float)
features_testing = lf.get_gauss_feat_multi(testing_image, sigma)
features_testing = features_testing.reshape((features_testing.shape[0], features_testing.shape[1]*features_testing.shape[2]))
labels = training_labels.ravel()


# Para cada píxel de la imagen nueva, encuentra cuál de los 1,000 grupos del entrenamiento es más similar
assignment_testing = kmeans.predict(features_testing)
probability_image = np.zeros((assignment_testing.size, nr_labels))

# Para cada píxel, ya sabemos a qué grupo pertenece
# Del entrenamiento sabemos las probabilidades de cada grupo
for l in range(nr_labels):
    probability_image[:, l] = cluster_probabilities[assignment_testing, l]
probability_image = probability_image.reshape(testing_image.shape + (nr_labels, ))



P_rgb = np.zeros(probability_image.shape[0:2]+(3, ))
k = min(nr_labels, 3)
P_rgb[:, :, :k] = probability_image[:, :, :k]
fig, ax = plt.subplots(1, 2)
ax[0].imshow(testing_image, cmap=plt.cm.gray)
ax[0].set_title('testing image')
ax[1].imshow(P_rgb)
ax[1].set_title('probabilities for testing image as RGB')
fig.tight_layout()
plt.show()

#%%
# SMOOTH PROBABILITY MAP PASO E

# Este código refina los resultados para obtener una segmentación más limpia y coherente.
# 


# El parámetro sigma = 3 controla cuánto suavizado aplicar: valores más altos = más suavizado pero menos detalle.
sigma = 3 # Gaussian smoothing parameter

seg_im_max = np.argmax(P_rgb, axis = 2)
c = np.eye(P_rgb.shape[2])
P_rgb_max = c[seg_im_max]

probability_smooth = np.zeros(probability_image.shape)
for i in range(0, probability_image.shape[2]):
    probability_smooth[:, :, i] = scipy.ndimage.gaussian_filter(probability_image[:, :, i], sigma, order=0)
seg_im_smooth = np.argmax(probability_smooth, axis=2)

probability_smooth_max = c[seg_im_smooth]

P_rgb_smooth = np.zeros(probability_smooth_max.shape[0:2]+(3, ))
k = min(nr_labels, 3)
P_rgb_smooth[:, :, :k] = probability_smooth[:, :, :k]
P_rgb_smooth_max = np.zeros(probability_smooth_max.shape[0:2]+(3, ))
P_rgb_smooth_max[:, :, :k] = probability_smooth_max[:, :, :k]

# Display result
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
ax[0][0].imshow(P_rgb[:, :, 0])
ax[0][1].imshow(P_rgb[:, :, 1])
ax[0][2].imshow(P_rgb[:, :, 2])
ax[0][3].imshow(P_rgb_max)
ax[1][0].imshow(P_rgb_smooth[:, :, 0])
ax[1][1].imshow(P_rgb_smooth[:, :, 1])
ax[1][2].imshow(P_rgb_smooth[:, :, 2])
ax[1][3].imshow(P_rgb_smooth_max)
fig.tight_layout()
plt.show()


# Sin suavizar:

# Puede tener píxeles aislados mal clasificados
# Bordes irregulares
# "Sal y pimienta" (ruido)

# Con suavizar:

# Regiones más homogéneas
# Bordes más suaves
# Elimina clasificaciones inconsistentes
# %%
