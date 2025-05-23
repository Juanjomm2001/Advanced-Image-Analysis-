# %% [markdown]
# ## 7.4-limeshone_solution

# %%
import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder
np.bool = bool

RGB = skimage.io.imread('./week7/rammed-earth-layers-limestone.jpg').astype(np.int32)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(RGB)
I = np.mean(RGB, axis=2)
ax[0, 0].set_title('Input')

#%% SETTINGS FOR GEOMETRIC CONSTRAINS
delta = 1 # smoothness very constrained, try also 3 to see less smoothness

#%% DARKEST LINE
layer = slgbuilder.GraphObject(I) #Crea una superficie donde el coste es la intensidad de la imagen
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)
helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1, :], axis=0) - 1
ax[0, 1].imshow(RGB)
ax[0, 1].plot(segmentation_line, 'r')
ax[0, 1].set_title('Darkest line')



#%% TWO DARK LINES
layers = [slgbuilder.GraphObject(I),slgbuilder.GraphObject(I)]
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)
helper.add_layered_containment(layers[0], layers[1], min_margin=50, max_margin=200)

helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1, :], axis=0) - 1 for s in segmentations]

ax[1, 0].imshow(RGB)
for line in segmentation_lines:
    ax[1,0].plot(line, 'r')
ax[1, 0].set_title('Two dark lines')



#%% DARKEST REGION
layers = [slgbuilder.GraphObject(0*I), slgbuilder.GraphObject(0*I)] # no on-surface cost
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)
helper.add_layered_boundary_cost()
helper.add_layered_region_cost(layers[0], 255 - I, I)
helper.add_layered_region_cost(layers[1], I, 255 - I)
helper.add_layered_smoothness(delta=delta, wrap=False)  
helper.add_layered_containment(layers[0], layers[1], min_margin=1, max_margin=200)

helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1, :], axis=0) - 1 for s in segmentations]

ax[1, 1].imshow(RGB)
for line in segmentation_lines:
    ax[1, 1].plot(line, 'r')
ax[1, 1].set_title('Dark region')
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7.5  DENTAL SOLUTION
# 7.5 Exercise: Quantifying dental tomograms
# 
# Meta: Medir cuánto contacto hay entre el implante y el hueso circundante.
# 

# %%
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import slgbuilder
np.bool = bool


# %% [markdown]
# # Investigate data

# %%
V = skimage.io.imread('./week7/dental_slices.tif')
fig, ax = plt.subplots(1, len(V), figsize=(10, 5))
for a, v in zip(ax, V):
    a.imshow(v, cmap='gray')
    a.axis('off')

# %% [markdown]
# ## Unfold, solve, and quantify

# %%
fig, ax = plt.subplots(2, len(V), figsize=(12, 6))

# Setttings for unfolding.
a = 180  # number of angles for unfolding
angles = np.linspace(0, 2 * np.pi, a, endpoint=False)  # angular coordinate
center = (np.array(V.shape[1:]) - 1) / 2
radii = np.arange(min(V.shape[1:]) / 2) + 1  # radial coordinate for unwrapping
X = center[0] + np.outer(radii, np.cos(angles))
Y = center[1] + np.outer(radii, np.sin(angles))
grid = np.stack((Y.ravel(), X.ravel()), axis=1)

#  Settings for cut
delta = 2

for i, I in enumerate(V):
    
    # Unfolding 
    F = scipy.interpolate.RectBivariateSpline(np.arange(I.shape[0]), np.arange(I.shape[1]), I)
    #val = F(snake[:, 0], snake[:, 1], grid=False)
    U = F(grid[:, 0], grid[:, 1], grid=False).reshape((len(radii), a))
    
    # Cost, making it int32
    cost = np.diff(U.astype(float), axis=0)
    cost = np.minimum(cost, 0) 
    cost -= cost.min()
    cost /= cost.max()
    cost *= (2 ** 16) - 1
    cost = cost.astype(np.int32)

    # Cut
    layer = slgbuilder.GraphObject(cost)
    helper = slgbuilder.MaxflowBuilder()
    helper.add_object(layer)
    helper.add_layered_boundary_cost()
    helper.add_layered_smoothness(delta=delta, wrap=True)
    
    helper.solve()
    segmentation = helper.what_segments(layer)
    r0 = segmentation.sum(axis=0) - 0.5 # not 1 because diff 
       
    # Expanding 20 pixels and folding back
    r20 = r0 + 20
    x20 = center[0] + r20 * np.cos(angles)
    y20 = center[1] + r20 * np.sin(angles)
    
    S = (I > 200).astype(int) + (I > 110).astype(int)
    ax[0, i].imshow(S)
    ax[0, i].plot(x20, y20, 'r')

    ax[1, i].imshow(U)
    ax[1, i].set_aspect(a/len(radii))
    ax[1, i].plot(r20, 'r')
    
    # Getting contact values
    contact_from_image = (F(y20, x20, grid=False) > 110).mean()
    ax[0, i].set_title(f'Bone contact:\n {contact_from_image:0.3g}')
    contact_from_unfolded = (U[r20.astype(int), np.arange(len(r20))] > 110).mean()
    ax[1, i].set_title(f'Bone contact:\n {contact_from_unfolded:0.3g}')

    
fig.tight_layout()
plt.show()


# %%



