import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pylab as pl
from sklearn.metrics import confusion_matrix,accuracy_score
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler

def get_train_data(is_train=True):
    data_path = "../train"
    num_cat = 25
    image_files = []
    labels = []

    work_folders = [int(f) for f in os.listdir(data_path) if
                    os.path.isdir(os.path.join(data_path, f)) and '.svn' not in f]
    work_folders.sort()
    work_folders = work_folders[:num_cat]
    for cat in work_folders:
        cat = str(cat)
        cat_folder = os.path.join(data_path, cat)
        images = [f for f in os.listdir(cat_folder) if os.path.isfile(os.path.join(cat_folder, f)) and '.svn' not in f]
        #images = images[:-50] if is_train else images[-50:]
        images = images[0:100] if is_train else images[100:200]

        for img in images:
            img_file = os.path.join(cat_folder, img)
            image_files.append(img_file)
            labels.append(int(cat))

    return image_files, labels

train_image_files, train_image_labels = get_train_data(True)
test_image_files, test_image_labels = get_train_data(False)
print('Done reading images')

def feature_extract(image_files):
    orb = cv2.ORB_create()
    des_list = []
    for image_path in image_files:
        im = cv2.imread(image_path)
        if im is not None:
            kp = orb.detect(im, None)
            keypoints, descriptor = orb.compute(im, kp)
            des_list.append((image_path, descriptor))

    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    descriptors_float = descriptors.astype(float)

    return descriptors_float, des_list

def testing(model, test_image_descriptors, test_image_descriptors_list, test_image_labels):
    test_features = np.zeros((len(test_image_labels), k), "float32")
    for i in range(len(test_image_labels)):
        words, distance = vq(test_image_descriptors_list[i][1], voc)
        for w in words:
            test_features[i][w] += 1

    test_features = stdslr.transform(test_features)
    prediction = model.predict(test_features)
    accuracy = accuracy_score(test_image_labels, prediction)
    print(accuracy)

train_image_descriptors, train_image_descriptors_list = feature_extract(train_image_files)
test_image_descriptors, test_image_descriptors_list = feature_extract(test_image_files)
print('Done feature extaction of images')


# KMEAN
k=200
voc,variance=kmeans(train_image_descriptors,k,1)
im_features=np.zeros((len(train_image_files), k),"float32")
for i in range(len(train_image_files)):
    words,distance=vq(train_image_descriptors_list[i][1],voc)
    for w in words:
        im_features[i][w]+=1


stdslr=StandardScaler().fit(im_features)
im_features=stdslr.transform(im_features)

from sklearn.svm import LinearSVC
clf=LinearSVC(max_iter=5)
clf.fit(im_features,np.array(train_image_labels))
print('Done training of images')


testing(clf, test_image_descriptors, test_image_descriptors_list, test_image_labels)


