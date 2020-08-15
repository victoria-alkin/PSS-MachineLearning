import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans

# Read data from file and create dataframe
labels_file = open(r"C:\Users\victoria.alkin\Documents\Labeling Meshes 2\labels.pkl","rb")
labels_db = pickle.load(labels_file)
labels_file.close()
labels_df = pd.DataFrame(labels_db)
strange_points_df = pd.DataFrame()

for i in labels_df.index:
    if (labels_df.loc[i,'label']=='2' and labels_df.loc[i,'x']>5.5 and labels_df.loc[i,'y']>2):
        strange_points_df = strange_points_df.append(labels_df.loc[i])

strange_points_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Strange Points\strange_points.csv")
strange_points_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Strange Points\strange_points.pkl","wb")
pickle.dump(strange_points_df, strange_points_file)
strange_points_file.close()