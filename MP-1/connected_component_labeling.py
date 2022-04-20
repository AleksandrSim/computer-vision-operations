# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def connected_component_labeling(image, threshhold=60):
    height, width = len(image), len(image[0])

    target = [[0] * width for i in range(height)]

    lis = [1]
    for row in range(1, height):
        for column in range(1, width):
            if image[row][column] != 0:

                if target[row][column - 1] != 0 and (target[row - 1][column]) == target[row][column - 1]:
                    target[row][column] = target[row][column - 1]

                elif target[row][column - 1] != 0 and target[row - 1][column] == 0:
                    target[row][column] = target[row][column - 1]


                elif target[row - 1][column] != 0 and target[row][column - 1] == 0:
                    target[row][column] = target[row - 1][column]


                elif (target[row - 1][column] != 0) and (target[row][column - 1] != 0) and (
                        target[row - 1][column] != target[row][column - 1]):
                    target[row][column] = target[row][column - 1]
                    value_to_keep = target[row][column - 1]
                    value_to_change = target[row - 1][column]
                    if value_to_change in lis:
                        lis.remove(value_to_change)

                    for row_2 in range(1, height):
                        for column_2 in range(1, width):
                            if target[row_2][column_2] == value_to_change:
                                target[row_2][column_2] = value_to_keep



                elif (target[row - 1][column] == target[row][column - 1]) and (target[row - 1][column] == 0):
                    value = lis[-1]

                    target[row][column] = value

                    lis.append(value + 1)

    dic = {}
    value = 20
    for i in lis:
        dic[i] = value
        value += 40

    for row in range(1, height):
        for column in range(1, width):
            if target[row][column] != 0:
                target[row][column] = dic[target[row][column]]

    t = 0

    for k in dic:
        count = 0
        for row in range(1, height):
            for column in range(1, width):
                if target[row][column] == dic[k]:
                    count += 1
        if count < threshhold:
            for row in range(1, height):
                for column in range(1, width):
                    if target[row][column] == dic[k]:
                        target[row][column] = 0

    return target


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = Image.open('face (2).bmp')
    image = np.asarray(image)
    new_image = connected_component_labeling(image, threshhold=100)
    plt.imshow(new_image)
    plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
