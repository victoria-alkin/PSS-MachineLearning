import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

#Read 160,000 data points
v3features_file = open(r"C:\Windows\System32\victoria\Training Set\V3features_embedded.pkl","rb")
v3features_db = pickle.load(v3features_file)
v3features_file.close()
v3features_df = pd.DataFrame(v3features_db,columns=['x','y'])
v3features = v3features_df.to_numpy()

# Read data from training set files and create dataframes
training_in_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_in_4.pkl","rb")
training_in_db = pickle.load(training_in_file)
training_in_file.close()
training_out_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_out_4.pkl","rb")
training_out_db = pickle.load(training_out_file)
training_out_file.close()
training_in_df = pd.DataFrame(training_in_db, columns=['x','y'])
training_out_df = pd.DataFrame(training_out_db, columns=['label'])
training_in = training_in_df.to_numpy()
training_out_column = training_out_df.to_numpy()
training_out = np.ravel(training_out_column)

# Read data from testing set files and create dataframes
testing_in_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_in_4.pkl","rb")
testing_in_db = pickle.load(testing_in_file)
testing_in_file.close()
testing_out_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_out_4.pkl","rb")
testing_out_db = pickle.load(testing_out_file)
testing_out_file.close()
testing_in_df = pd.DataFrame(testing_in_db, columns=['x','y'])
testing_out_df = pd.DataFrame(testing_out_db, columns=['label'])
testing_in = testing_in_df.to_numpy()
testing_out = testing_out_df.to_numpy()

# Fit SVM model
C=1.0
linsvc = svm.LinearSVC(C=C).fit(training_in, training_out)
svclink = svm.SVC(kernel='linear', C=C).fit(training_in, training_out)
svcrbfk = svm.SVC(kernel='rbf', C=C).fit(training_in, training_out)
svcpolyk = svm.SVC(kernel='poly', degree=3, C=C).fit(training_in, training_out)
# svcrbfk = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(training_in, training_out)

classnames = ["spine", "shaft", "soma", "proximal process"]
# Plot linear SVC confusion matrix
np.set_printoptions(precision=2)
titles_options = [("Linear SVC, without normalization", None), ("Linear SVC, normalized", "true")]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(linsvc,testing_in,testing_out,cmap=plt.cm.Blues, display_labels=classnames, normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

# Plot SVC with linear kernel confusion matrix
np.set_printoptions(precision=2)
titles_options = [("SVC with Linear Kernel, without normalization", None), ("SVC with Linear Kernel, normalized", "true")]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svclink,testing_in,testing_out,cmap=plt.cm.Blues, display_labels=classnames, normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

# Plot SVC with RBF kernel confusion matrix
np.set_printoptions(precision=2)
titles_options = [("SVC with RBF Kernel, without normalization", None), ("SVC with RBF Kernel, normalized", "true")]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svcrbfk,testing_in,testing_out,cmap=plt.cm.Blues, display_labels=classnames, normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

# Plot SVC with polynomial kernel confusion matrix
np.set_printoptions(precision=2)
titles_options = [("SVC with Polynomial Kernel, without normalization", None), ("SVC with Polynomial Kernel, normalized", "true")]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svcpolyk,testing_in,testing_out,cmap=plt.cm.Blues, display_labels=classnames, normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


figure = plt.figure()
# create a mesh to plot in
h = 0.2
x_min, x_max = testing_in[:,0].min() - 1, testing_in[:,0].max() + 1
y_min, y_max = testing_in[:,1].min() - 1, testing_in[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['Linear SVC', 'SVC with Linear Kernel', "SVC with RBF Kernel", "SVC with Polynomial (Degree 3) Kernel"]

for i, clf in enumerate((linsvc, svclink, svcrbfk, svcpolyk)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8, figure = figure)

    # Plot also the training points
    plt.scatter(v3features[:,0], v3features[:,1],s=0.5,alpha=0.5,figure=figure)
    #plt.scatter(testing_in[:,0], testing_in[:, 1], cmap=plt.cm.coolwarm, figure = figure)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()