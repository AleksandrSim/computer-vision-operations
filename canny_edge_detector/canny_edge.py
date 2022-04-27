import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os


def convert_to_gray(image, red=0.2989, green=0.5870, blue=0.1140):
    target = [[0] * len(image[0]) for i in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            target[i][j] = int(red * image[i][j][0]) + int(green * image[i][j][1]) + int(blue * image[i][j][2])
    return target

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def generate_kernel(size=5, sig=3, n=1):
    if size % 2 == 0:
        size += 1
    raw_kernel = np.array([[0] * size for i in range(size)], dtype="float32")
    center = len(raw_kernel) // 2
    for i in range(size):
        for j in range(size):
            raw_kernel[i][j] = gaussian(i, j, center, sig)
    overall_sum = np.sum(raw_kernel)
    coef = 1 / overall_sum

    raw_kernel = raw_kernel * coef

    return raw_kernel



def gaussian(x,y, center =1, nsig=0.4):
    return (1/np.sqrt(2*nsig**2*math.pi))*math.e**(-(((abs(x-center)**2)+(abs(y-center)**2))/(2*nsig**2)))


def filter_image(image, kernel_size=3, sigm=0.4):
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2
    length, width = len(image), len(image[0])

    image = np.pad(image, pad_width=center)

    kernel = generate_kernel(kernel_size, sigm)
    new_image = np.zeros((length,width))

    for row in range(center, length+1):
        for column in range(center, width+1):
            new_image[row - center, column - center] = np.sum(
                np.dot(image[row - 1:row + 2, column - 1:column + 2], kernel))

    return new_image


def filter_image_sobel(image):
    center = 1

    length, width = len(image), len(image[0])
    image = np.pad(image, pad_width=1)

    x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    new_image_X = np.zeros((length,width))
    new_image_Y =np.zeros((length,width))

    new_image = np.zeros((length,width))
    for row in range(1, length+1):
        for column in range(1, width+1):
            #           new_image_X[row-center,column-center]= np.sum(np.dot(x, image[row-1:row+2, column-1:column+2]))
            #           new_image_Y[row-center,column-center]= np.sum(np.dot(image[row-1:row+2, column-1:column+2],y, ))
            new_image_X[row - center, column - center] = np.sum(image[row - 1:row + 2, column - 1:column + 2] * x)

            new_image_Y[row - center, column - center] = np.sum(image[row - 1:row + 2, column - 1:column + 2] * y)
            new_image[row - center, column - center] = np.sqrt(
                new_image_X[row - center, column - center] ** 2 + new_image_Y[row - center, column - center] ** 2)

    theta = np.arctan2(new_image_Y, new_image_X)
    #    theta = np.arctan2(new_image_X, new_image_Y)
    new_image = new_image / np.max(new_image) * 255

    return new_image, theta


def threshold(img, lowThreshold=0.06, highThreshold=0.11):
    highThreshold = np.max(img) * highThreshold
    lowThreshold = lowThreshold * highThreshold

    length, width = len(img), len(img[0])
    new_image = np.zeros((length, width), dtype=np.int32)

    for i in range(length):
        for j in range(width):

            if img[i, j] >= highThreshold:
                new_image[i, j] = 255

            elif img[i, j] <= lowThreshold:
                new_image[i, j] = 0


            else:
                new_image[i, j] = 25

    return new_image



def edge_linking(img, weak=25, strong=255):
    length, width = len(img), len(img[0])
    for i in range(1, length):
        for j in range(1, width):
            if (img[i,j] == weak):
                breaker =False
                for upper in range(-1,2):
                    if breaker == True:
                        break
                    for horiz in range(-1,2):
                        if (img[i+upper, j+horiz])== strong:
                            braker = True
                            img[i,j]= strong
                            break
                if img[i,j]!= strong:
                    img[i][j] =0


    return img
def non_max_suppression(img, D):
    length, width = len(img), len(img[0])
    new_image = np.zeros((length, width), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, length - 1):
        for j in range(1, width - 1):
            prev, after = 255, 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                prev = img[i, j + 1]
                after = img[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                prev = img[i + 1, j - 1]
                after = img[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                prev = img[i + 1, j]
                after = img[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                prev = img[i - 1, j - 1]
                after = img[i + 1, j + 1]

            if (img[i, j] >= prev) and (img[i, j] >= after):
                new_image[i, j] = img[i, j]
            else:

                new_image[i, j] = 0

    return new_image
def show(values):
    plt.imshow(values)
    plt.show()


if __name__=='__main__':

    image_name = 'test1.bmp'
    if not os.path.isdir(image_name[:-4]):
        os.mkdir(image_name[:-4])
    image = np.array(Image.open(image_name))
    show(image)
    image = np.array(convert_to_gray(image))

    gaussian_image = filter_image(image, kernel_size=3, sigm=0.4)

    show(gaussian_image)
    new_image, theta = filter_image_sobel(gaussian_image)
    show(new_image)
    plt.hist(new_image.reshape(len(new_image[0])* len(new_image)), cumulative=True)
    plt.show()
    image_non_max = non_max_suppression(new_image, theta)

    show(image_non_max)
    new_img = threshold(image_non_max,lowThreshold=0.04, highThreshold=0.1)

    show(new_img)
    plt.imshow(edge_linking(new_img, weak =25), cmap='gray', vmin=0, vmax=255)
    plt.show()







