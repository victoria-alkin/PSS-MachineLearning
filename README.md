# PSS-MachineLearning
Using machine learning models to classify post-synaptic shapes based on 2D UMAP data.

MakingSets.py - creates pkl files with training and testing sets (selected using kmeans and random selection)

Sets.py - creates dataframes for training and testing input and output with 3 classes (spine, shaft, soma), 4 classes (spine, shaft, soma, proximal process), and 8 classes (spine, shaft, soma, proximal process, partial spine, partial shaft, merged spine, merged shaft)

3Classes.py, 4Classes.py, 8Classes.py - each python file trains four classifiers (linearSVC, SVM with linear kernel, SVM with RBF kernel, and SVM with polynomial kernel) and generates confusion matrices and decision boundaries for all four. The 3Classes file uses only the spine, shaft, and soma classes. The 4 classes file also includes proximal processes. The 8 classes file also includes partial and merged spines and shafts.
