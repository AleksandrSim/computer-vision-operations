import os
from PIL import Image
import numpy as np
import cv2




def convert_to_gray(image, red=0.2989, green=0.5870, blue=0.1140):
    target = [[0] * len(image[0]) for i in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            target[i][j] = int(red * image[i][j][0]) + int(green * image[i][j][1]) + int(blue * image[i][j][2])
    return target
def compute_mean_squarred(reference_image, target_image):
    errors = []
    length, width = reference_image.shape

    for i in range(length):
        for j in range(width):
            errors.append((reference_image[i][j] - target_image[i][j]) ** 2)
    #            errors.append((reference_image[i][j][1]-  target_image[i][j][1])**2)
    #           errors.append((reference_image[i][j][2]-  target_image[i][j][2])**2)
    overall_errors = sum(errors) / len(errors)

    return overall_errors

def compute_mean_squarred_2(reference_image, target_image):
    overall_errors = np.mean((reference_image - target_image)**2)

    return overall_errors



def cross_correlation(reference_image, target_image):
    negative_mean =-1* np.mean(np.corrcoef(reference_image.flatten(), target_image.flatten()))

    return negative_mean


def cross_correlation_normalized(reference_image, target_image):
    flatten_image_reference = reference_image.flatten()
    target_image_flatten = target_image.flatten()
    negative_mean =-1* np.mean(np.corrcoef(flatten_image_reference-flatten_image_reference.mean(), target_image_flatten - target_image_flatten.mean()))

    return negative_mean




def iterate_over_target_image(reference_image, target_image, previous_image, method='normalized_cross_correlation', prev = True):

    factor = 13
    target_image = np.array(convert_to_gray(target_image))
    length,width = target_image.shape

    kernel_width = 4 * factor
    kernel_length = 3 * factor
    width_stop = width - kernel_width
    length_stop = length - kernel_length
    target = float('inf')
    if method =='normalized_cross_correlation':
        for i in range(5,length_stop-5):
            for j in range(25, width_stop-5):
                current_mean = cross_correlation_normalized(reference_image, target_image[i:i + kernel_length, j:j + kernel_width])
                if 'prev'==True:
                    current_mean_2 = cross_correlation_normalized(previous_image, target_image[i:i + kernel_length, j:j + kernel_width])
                    current_mean = min(current_mean, current_mean_2)

                if current_mean < target:
                    target = current_mean
                    coordinates = i, j

        return coordinates, target_image[i:i + kernel_length, j:j + kernel_width]
    elif method =='lse':
        for i in range(5,length_stop-15):
            for j in range(5, width_stop-15):
                current_mean = compute_mean_squarred_2(reference_image, target_image[i:i + kernel_length, j:j + kernel_width])
                if 'prev'==True:
                    current_mean_2 = compute_mean_squarred_2(previous_image, target_image[i:i + kernel_length, j:j + kernel_width])
                    current_mean = min(current_mean, current_mean_2)

                if current_mean < target:
                    target = current_mean
                    coordinates = i, j

        return coordinates, target_image[i:i + kernel_length, j:j + kernel_width]






def save_video(reference_image, directory, kernel_length, kernel_width, path = 'image_girl'):
    writer = cv2.VideoWriter("normalized_cross_corr_local.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (128, 96))
    (x,y)=0,0
    for k, imag in enumerate(destinations):
        img = path + '/' + imag
        img = np.array(Image.open(img))
        coordinates, previous_image = iterate_over_target_image(reference_image, img,reference_image,method='normalized_cross_correlation', prev=True)
        x,y = coordinates
        print(k)
        #    img[x:x+kernel_length, y:y+kernel_width,:]
        cv2.rectangle(img, (y, x), (y + kernel_width, x + kernel_length), (36, 255, 12), 2)
        writer.write(img)
    writer.release()






if __name__ =='__main__':
    path ='image_girl'
    destinations = sorted(os.listdir(path))
    factor = 13
    kernel_width = 4*factor
    kernel_length = 3 *factor

    reference_image = np.array(Image.open(path + '/' + destinations[0]))
    length, width = reference_image.shape[:2]


    reference_image = np.array(convert_to_gray(reference_image[30:30 + kernel_length, 50:50 + kernel_width]))

    save_video(reference_image, destinations, kernel_length, kernel_width)






