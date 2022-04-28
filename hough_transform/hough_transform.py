import cv2 # to perform canny edge detection quickly
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image, ImageDraw
import os

from skimage.feature import peak_local_max
def convert_to_gray(image, red=0.2989, green=0.5870, blue=0.1140):
    target = [[0] * len(image[0]) for i in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            target[i][j] = int(red * image[i][j][0]) + int(green * image[i][j][1]) + int(blue * image[i][j][2])
    return target
def get_y(x,rho_param, theta_param):
    if theta_param !=0:
        return (-math.cos(theta_param) / math.sin(theta_param)) * x + (rho_param / math.sin(theta_param))

    else:
        rho_param

def params_to_coordinates(rho, theta, width):
    x1 = 0
    x2 = width
    y1 = int(get_y(0, rho, theta))
    y2 = int(get_y(width, rho, theta))
    return x1, y1, x2, y2




def accamulator_vectorized(edged_img,ang_res, dist_res):
    heigth, width = edged_img.shape
    max_rho_param = np.hypot(width, heigth)

    num_theta_param = int(math.pi / ang_res)
    num_rho_param = int(2 * max_rho_param / dist_res) - 1

    accum = np.zeros((num_rho_param, num_theta_param)).astype(np.int16)
    for column in range(0, heigth):
        for row in range(0, width):
            pixel = edged_img[column][row]
            if pixel != 0:
                slope_array = np.arange(0, math.pi, ang_res)
                rho_array = column * np.sin(slope_array)+ row * np.cos(slope_array)
                theta_index_arr = np.rint(slope_array / ang_res).astype(np.int32)
                rho_index_arr = np.rint(rho_array / dist_res + num_rho_param / 2).astype(np.int32)
                acc = np.bincount( theta_index_arr+rho_index_arr * accum.shape[1])
                acc.resize(accum.shape)
                accum += acc
    return accum,num_rho_param




def show(values):
    plt.imshow(values)
    plt.show()

def plot_hist(acc):
    new= acc.reshape(len(acc[0])*len(acc))
    plt.hist(new[new>0])
    plt.show()



def get_lines(acc, img, num_rho_param,distance_resolution, angle_resolution, width, threshold = 40):
    acc[acc<threshold]=0
    distances = 3
    img_array = np.array(img)
    locations = peak_local_max(acc, min_distance=distances)
    locations = locations[:5:]
    for rho_index, theta_index in locations:
        rho = (rho_index - num_rho_param / 2) * distance_resolution
        theta = theta_index * angle_resolution
        x1, y1, x2, y2 = params_to_coordinates(rho, theta,width)
        draw = ImageDraw.Draw(img)
        draw.line((x1, y1, x2, y2), fill=128)

    img_array = np.array(img)
    plt.imshow(img_array)
    plt.show()

    return img_array




if __name__=='__main__':
    DIST_RES = 0.5
    ANG_RES = math.pi/180/6
    THRESHOLD = 60
    image_name = 'input.bmp'
    if not os.path.isdir(image_name[:-4]):
        os.mkdir(image_name[:-4])
    image = np.array(Image.open(image_name))
    heigth, width= len(image), len(image[0])
    image_gray = np.array(convert_to_gray(image))
    edges = cv2.Canny(image, 50, 200, apertureSize=3)

    acc,num_rho_param =accamulator_vectorized(edges, math.pi/180/6,0.5)
    plt.imshow(acc, cmap='viridis')
    plt.show()
    plot_hist(acc)
    image = get_lines(acc, Image.open(image_name), num_rho_param,DIST_RES,ANG_RES, width,  threshold=40)










