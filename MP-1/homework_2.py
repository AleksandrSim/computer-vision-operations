import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def dillation(image, size_of_kernel=3):
    if size_of_kernel <= 2:
        size_of_kernel = 3
    if size_of_kernel % 2 == 0:
        size_of_kernel = size_of_kernel - 1

    kernel = [[True] * size_of_kernel for i in range(size_of_kernel)]
    image = np.pad(image, pad_width=1)
    length, width = len(image), len(image[0])
    step = size_of_kernel // 2
    output = np.array([[0] * (width - size_of_kernel + 1) for i in range(length - size_of_kernel + 1)])

    for row in range(1, length - step):
        for column in range(1, width - step):
            current_matrix = []

            for z in range(-1 * step, step + 1):
                current_matrix.append(list(image[row + z][column - 1:column + 2]))

            breakout_flag = False

            for i in range(len(current_matrix)):
                for j in range(len(current_matrix[0])):
                    if current_matrix[i][j] == True and current_matrix[i][j] == kernel[i][j]:
                        output[row - step][column - step] = kernel[i][j]
                        breakout_flag = True
                        break
                if breakout_flag == True:
                    break
                elif breakout_flag == False:
                    output[i - step][j - step] = 0

    return output


def errosion(image, size_of_kernel=3):
    if size_of_kernel <= 2:
        size_of_kernel = 3
    if size_of_kernel % 2 == 0:
        size_of_kernel = size_of_kernel - 1

    kernel = [[True] * size_of_kernel for i in range(size_of_kernel)]
    image = np.pad(image, pad_width=1)
    length, width = len(image), len(image[0])
    step = size_of_kernel // 2
    output = np.array([[0] * (width - size_of_kernel + 1) for i in range(length - size_of_kernel + 1)])

    current_matrix = []

    for row in range(1, length - step):
        for column in range(1, width - step):
            current_matrix = []

            for z in range(-1 * step, 1 * step + 1):
                current_matrix.append(list(image[row + z][column - step:column + step + 1]))

            if current_matrix == kernel:
                output[row - step][column - step] = 1



            else:
                output[row - step][column - step] = 0

    return output


def opening(image, errosion_kernel_size, dillation_kernel_size):
    errosed_image = errosion(image, errosion_kernel_size)
    dillated_image = dillation(errosed_image, dillation_kernel_size)
    return dillated_image


def closing(image, errosion_kernel_size, dillation_kernel_size):
    dillated_image = dillation(image, dillation_kernel_size)

    errosed_image = errosion(dillated_image, errosion_kernel_size)

    return errosed_image


def contour(image_A, image_B):
    image = image_A - image_B
    return image



if __name__ =='__main__':
    image = np.asarray(Image.open('gun_2_homework.bmp'))
    plt.imshow(dillation(image, size_of_kernel=7))
    plt.show()
    plt.imshow(errosion(image, size_of_kernel=3))
    plt.show()
    plt.imshow(opening(image, errosion_kernel_size = 3, dillation_kernel_size= 3))
    plt.show()
    plt.imshow(closing(image, errosion_kernel_size=7, dillation_kernel_size=7))
    plt.show()
    itermiediate = closing(image, errosion_kernel_size=7, dillation_kernel_size=7)

    plt.imshow(contour(dillation(itermiediate), itermiediate))
    plt.show()

