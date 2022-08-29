import pickle
import random
import numpy as np
import matplotlib.pyplot as plt   # save the figure and load the figure
import cv2  # image processing
import os


# at the end of address put / symbole
DataDir = r"E:\NDT\Academic\Sem_3\Project I\Testing\test_1\PetImages/"

CATEGORIES = ["Dog", "Cat"]

'''

for i in CATEGORIES:
    path = os.path.join(DataDir, i)
    for img in os.listdir(path):
        # @ grayscale is convart RGB to gray reduce size and easy to nural network
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break

img_size = 120

# resize images and store in new array
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap='gray')
plt.show()


training_data = []


def create_training_data():  # create traing data set and store it, remove corrpted iamges
    for i in CATEGORIES:
        path = os.path.join(DataDir, i)
        class_num = CATEGORIES.index(i)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])

            except Exception as e:
                pass


create_training_data()
print(len(training_data))

# shufful training data
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)


# give features to the nureal network
x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

print(x[0].reshape(-1, img_size, img_size, 1))

x = np.array(x).reshape(-1, img_size, img_size, 1)


# dump

pickle_out = open(
    r"E:\NDT\Academic\Sem_3\Project I\Testing\test_1\x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open(
    r"E:\NDT\Academic\Sem_3\Project I\Testing\test_1\y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

'''
