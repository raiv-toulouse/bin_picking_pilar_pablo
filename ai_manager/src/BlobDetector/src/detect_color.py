import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

# Read the image for blob detection
image_path = './final_pictures/img1610101287.71.png'
im = cv2.imread(image_path)
# plt.imshow(im)
# plt.show()

# Turning image into RGB
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# plt.imshow(im_rgb)
# plt.show()

# Visualize Image in RGB Color Space
r, g, b = cv2.split(im_rgb)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = im_rgb.reshape((np.shape(im_rgb)[0]*np.shape(im_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

# Visualize Imagen in HSV Color Space
im_hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(im_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

### Picking Out a Range

sensitivity = 30
lower_white = np.array([0, 0, 255-sensitivity])
upper_white = np.array([255, sensitivity, 255])

lw_square = np.full((10, 10, 3), lower_white, dtype=np.uint8) / 255.0
dw_square = np.full((10, 10, 3), upper_white, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(lw_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(dw_square))
plt.show()

# Threshold the HSV image to get only white colors
mask = cv2.inRange(im_hsv, lower_white, upper_white)
plt.imshow(mask, cmap="gray")

# To impose the mask on top of the original image, you can use cv2.bitwise_and()
# which keeps every pixel in the given image if the corresponding value in the mask is 1
result = cv2.bitwise_and(im_hsv, im_hsv, mask=mask)
plt.imshow(result)
plt.show()

light_white = (0, 0, 0)
dark_white = (0, 255, 0)
light_square = np.full((10, 10, 3), light_white, dtype=np.uint8)/255
dark_square = np.full((10, 10, 3), dark_white, dtype=np.uint8)/255
plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(light_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(dark_square))
plt.show()










