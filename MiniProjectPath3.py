#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list,images,labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)

  image_list = []
  labels_list = []
  for i in range(len(labels)):
      if labels[i] in number_list:
          image_list.append(images[i])
          labels_list.append(labels[i])

  return np.array(image_list), np.array(labels_list)

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 
  plt.figure(figsize=(10, 4))
  for i in range(len(images)):
      plt.subplot(1, len(images), i + 1)
      plt.imshow(images[i], cmap='gray')
      plt.title(str(labels[i]))
      plt.axis('off')
  plt.show()

def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  correct = 0
  for i in range(len(results)):
      if results[i] == actual_values[i]:
          correct += 1

  Accuracy = correct / len(results)
  return Accuracy

#Part 1
class_numbers = [2, 0, 8, 7, 5]
class_number_images, class_number_labels = dataset_searcher(class_numbers, images, labels)


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
#so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
#the new shape would be 718 by 64

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)

#Part 3 Calculate model1_results using model_1.predict()
model1_results = model_1.predict(X_test_reshaped)

# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)

allnumbers_images_reshaped = allnumbers_images.reshape(allnumbers_images.shape[0], -1)
#print_numbers(allnumbers_images, model_1.predict(allnumbers_images_reshaped))
#Part 2
print_numbers(allnumbers_images[:10], model_1.predict(allnumbers_images_reshaped[:10]))

#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)

model2_results = model_2.predict(X_test_reshaped)
model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall results of the KNN model is " + str(model2_Overall_Accuracy))


#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0,max_iter=1000)
model_3.fit(X_train_reshaped, y_train)

model3_results = model_3.predict(X_test_reshaped)
model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall results of the MLP model is " + str(model3_Overall_Accuracy))



#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison
X_train_poison_reshaped = X_train_poison.reshape(X_train.shape[0], -1)


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train

#Gaussian
model_1_poison = GaussianNB()
model_1_poison.fit(X_train_poison_reshaped, y_train)
print("Gaussian poisoned accuracy:", OverallAccuracy(model_1_poison.predict(X_test_reshaped), y_test))

#KNN
model_2_poison = KNeighborsClassifier(n_neighbors=10)
model_2_poison.fit(X_train_poison_reshaped, y_train)
print("KNN poisoned accuracy:", OverallAccuracy(model_2_poison.predict(X_test_reshaped), y_test))

#MLP
model_3_poison = MLPClassifier(random_state=0, max_iter=1000)
model_3_poison.fit(X_train_poison_reshaped, y_train)
print("MLP poisoned accuracy:", OverallAccuracy(model_3_poison.predict(X_test_reshaped), y_test))


#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

kpca = KernelPCA(n_components=64, kernel='rbf', fit_inverse_transform=True, gamma=0.01)

X_train_denoised = kpca.fit_transform(X_train_poison_reshaped)
X_train_denoised = kpca.inverse_transform(X_train_denoised)


#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.

#Gaussian
model_1_den = GaussianNB()
model_1_den.fit(X_train_denoised, y_train)
print("Gaussian denoised accuracy:", OverallAccuracy(model_1_den.predict(X_test_reshaped), y_test))

#KNN
model_2_den = KNeighborsClassifier(n_neighbors=10)
model_2_den.fit(X_train_denoised, y_train)
print("KNN denoised accuracy:", OverallAccuracy(model_2_den.predict(X_test_reshaped), y_test))

#MLP
model_3_den = MLPClassifier(random_state=0, max_iter=1000)
model_3_den.fit(X_train_denoised, y_train)
print("MLP denoised accuracy:", OverallAccuracy(model_3_den.predict(X_test_reshaped), y_test))