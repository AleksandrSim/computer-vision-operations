from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv
from sklearn.mixture import GaussianMixture

def extract_colors(image):
    color_1 = [[0] * len(image[0]) for i in range(len(image))]
    color_2 = [[0] * len(image[0]) for i in range(len(image))]
    color_3 = [[0] * len(image[0]) for i in range(len(image))]
    for i in range(len(image)):
        for j in range(0, len(image[0])):
            color_1[i][j] = image[i][j][0]
            color_2[i][j] = image[i][j][1]
            color_3[i][j] = image[i][j][2]

    plt.hist(color_1)
    plt.show()
    plt.hist(color_2)
    plt.show()
    plt.hist(color_3)
    plt.show()
def values_for_each_channel(original_image,image):
    all_the_positions = []
    original_image_positions = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            res = []
            res.append(image[i][j][0])
            res.append(image[i][j][1])
            res.append(image[i][j][2])
            all_the_positions.append(res)
            res = []
            res.append(original_image[i][j][0])
            res.append(original_image[i][j][1])
            res.append(original_image[i][j][2])
            original_image_positions.append(res)
    return  all_the_positions, original_image_positions

def segment_colors_clusters(all_the_positions, original_image_positions, length, width,gaussian = False):
    if gaussian == False:
        kmeans = KMeans(n_clusters=5, random_state=0).fit(all_the_positions)
        predictions = kmeans.predict(all_the_positions)
        predicted = kmeans.predict(all_the_positions)
    else:
        gaussian = GaussianMixture(n_components=5).fit(all_the_positions)
        predictions = gaussian.predict(all_the_positions)
        predicted =gaussian.predict(all_the_positions)
    for j in set(predicted):
        segmented_image =[]
        for i in range(len(all_the_positions)):
            if predictions[i] == j:
                segmented_image.append(original_image_positions[i][0:3])
            else:
                segmented_image.append([0, 0, 0])
        plt.imshow(np.array(segmented_image).reshape(length, width, 3))
        plt.show()

def segment_colors_histogram_threshold(all_the_positions, original_image_positions):
        new1 = []
        for i in range(len(all_the_positions)):
            if all_the_positions[i][0] < 0.13 and (
                    all_the_positions[i][1] > 0.12 and all_the_positions[i][1] < 0.8) and (
                    all_the_positions[i][2] > 0.30 and all_the_positions[i][2] < 0.8):

                new1.append(original_image_positions[i][0:3])
            else:
                new1.append([0, 0, 0])

        plt.imshow(np.array(new1).reshape(length, width, 3))
        plt.show()


if __name__=='__main__':
    original_image = np.array(Image.open('baby.jpg'))
    image = rgb2hsv(original_image)
    extract_colors(original_image)
    length, width = len(image), len(image[0])
    all_the_positions, original_image_positions = values_for_each_channel(original_image,image)
    segment_colors_histogram_threshold(all_the_positions, original_image_positions)
    segment_colors_clusters(all_the_positions, original_image_positions, length, width, gaussian=True)


















