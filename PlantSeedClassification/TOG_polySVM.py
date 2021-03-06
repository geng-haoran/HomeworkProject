import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import *
from skimage.feature import hog
import sklearn.svm as svm
##Combining the train and test data
#get_ipython().system('cp -r /content/mnist_png/testing/* /content/mnist_png/training/')
#get_ipython().system('rm -rf /content/mnist_png/testing /content/mnist_png.tar.gz')


#Preparing the dataset 
path = '/data2/haoran/HW/HomeworkProject/PlantSeedClassification/datav3/eval'
image_path = []
# labels = []
for i in range(10):
  dir = os.path.join(path, LABEL2NAME[i])
  for file in os.listdir(dir):
    image_path.append(os.path.join(dir, file))
    # labels.append(i)
 
# print(len(image_path))
# exit(123)
def main(thresh):

  t0 = time.time()


  def CalcFeatures(img, th):
    sift = cv2.xfeatures2d.SIFT_create(th)
    kp, des = sift.detectAndCompute(img, None)
    return des
  
  '''
  All the files appended to the image_path list are passed through the
  CalcFeatures functions which returns the descriptors which are 
  appended to the features list and then stacked vertically in the form
  of a numpy array.
  '''

  features = []
  for file in image_path:
    img = cv2.imread(file, 0)
    img_des = CalcFeatures(img, thresh)
    if img_des is not None:
      features.append(img_des)
  features = np.vstack(features)

  '''
  K-Means clustering is then performed on the feature array obtained 
  from the previous step. The centres obtained after clustering are 
  further used for bagging of features.
  '''

  k = 150
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
  flags = cv2.KMEANS_RANDOM_CENTERS
  compactness, labels, centres = cv2.kmeans(features, k, None, criteria, 10, flags)

  '''
  The bag_of_features function assigns the features which are similar
  to a specific cluster centre thus forming a Bag of Words approach.  
  '''

  def bag_of_features(features, centres, k = 500):
      vec = np.zeros((1, k))
      for i in range(features.shape[0]):
          feat = features[i]
          diff = np.tile(feat, (k, 1)) - centres
          dist = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
          idx_dist = dist.argsort()
          idx = idx_dist[0]
          vec[0][idx] += 1
      return vec

  labels = []
  vec = []
  ppc = 16
  hog_images = []
  hog_features = []
  for file in image_path:
    img = cv2.imread(file, 0)
    # fd,hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2')
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(4, 4),block_norm= 'L2', visualize=True)
    # print(fd)
    # print(fd.shape)
    # if img_des is not None:
    vec.append(fd)
    labels.append(NAME2LABEL[file.split("/")[-2]])
  # print(np.array(vec).shape)
  # exit(123)

    
  '''
  Splitting the data formed into test and split data and training the 
  SVM Classifier.
  '''
  # print(len(vec))
  # print(np.array(vec).shape)
  # print(np.array(labels).shape)
  X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.2)
#   print(len(X_train).shape,len(X_test).shape,y_train.shape,len(y_test))
  clf = svm.SVC(kernel = 'poly')
  clf.fit(X_train, y_train)
  preds = clf.predict(X_test)
  acc = accuracy_score(y_test, preds)
  conf_mat = confusion_matrix(y_test, preds)

  t1 = time.time()
  
  return acc*100, conf_mat, (t1-t0)


accuracy = []
timer = []
for i in range(5,26,5):
  print('\nCalculating for a threshold of {}'.format(i))
  data = main(i)
  accuracy.append(data[0])
  conf_mat = data[1]
  timer.append(data[2])
  print('\nAccuracy = {}\nTime taken = {} sec\nConfusion matrix :\n{}'.format(data[0],data[2],data[1]))
