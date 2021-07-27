import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = "Desktop\\5.jpg"

vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

img = cv2.imread(PATH)
# img = plt.imread('Desktop\\5.jpg')
n, m, d = img.shape

edges_img = img.copy()

for row in range(3, n-2):
    for col in range(3, m-2):
        local_pixels = img[row-1:row+2, col-1:col+2, 0]

        vertical_transformed_pixels = vertical_filter*local_pixels
        vertical_score = vertical_transformed_pixels.sum()/4

        horizontal_transformed_pixels = horizontal_filter*local_pixels
        horizontal_score = horizontal_transformed_pixels.sum()/4

        edge_score = (vertical_score**2 + horizontal_score**2)**.5
        edges_img[row, col] = [edge_score]*3


edges_img = edges_img/edges_img.max()
print(edges_img)

# def save_image(name, image, path):
#     path_for_image = '{}\{}.jpg'.format(path, name)
#     cv2.imwrite(path_for_image, image)


cv2.imshow("RESULT", edges_img)

# plt.imshow(edges_img, cmap="gray")
# plt.show()
