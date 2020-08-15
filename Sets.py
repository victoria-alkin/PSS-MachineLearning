import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn

# Read data from training set file and create dataframes
training_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning\training.pkl","rb")
training_db = pickle.load(training_file)
training_file.close()
training_in_df = pd.DataFrame(training_db, columns=['x','y'])
training_out_df = pd.DataFrame(training_db, columns=['label'])

# Read data from training set file and create dataframes
testing_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning\testing.pkl","rb")
testing_db = pickle.load(testing_file)
testing_file.close()
testing_in_df = pd.DataFrame(testing_db, columns=['x','y'])
testing_out_df = pd.DataFrame(testing_db, columns=['label'])

# 3 CLASSES
training_in_3classes_df = training_in_df
training_out_3classes_df = training_out_df
testing_in_3classes_df = testing_in_df
testing_out_3classes_df = testing_out_df
for i in training_out_df.index:
    if (training_out_df.loc[i,'label']=='4' or training_out_df.loc[i,'label']=='5' or training_out_df.loc[i,'label']=='6' or training_out_df.loc[i,'label']=='7' or training_out_df.loc[i,'label']=='8' or training_out_df.loc[i,'label']=='9'):
        training_out_3classes_df = training_out_3classes_df.drop([i])
        training_in_3classes_df = training_in_3classes_df.drop([i])
for i in testing_out_df.index:
    if (testing_out_df.loc[i,'label']=='4' or testing_out_df.loc[i,'label']=='5' or testing_out_df.loc[i,'label']=='6' or testing_out_df.loc[i,'label']=='7' or testing_out_df.loc[i,'label']=='8' or testing_out_df.loc[i,'label']=='9'):
        testing_out_3classes_df = testing_out_3classes_df.drop([i])
        testing_in_3classes_df = testing_in_3classes_df.drop([i])

# 4 CLASSES
training_in_4classes_df = training_in_df
training_out_4classes_df = training_out_df
testing_in_4classes_df = testing_in_df
testing_out_4classes_df = testing_out_df
for i in training_out_df.index:
    if (training_out_df.loc[i,'label']=='5' or training_out_df.loc[i,'label']=='6' or training_out_df.loc[i,'label']=='7' or training_out_df.loc[i,'label']=='8' or training_out_df.loc[i,'label']=='9'):
        training_out_4classes_df = training_out_4classes_df.drop([i])
        training_in_4classes_df = training_in_4classes_df.drop([i])
for i in testing_out_df.index:
    if (testing_out_df.loc[i,'label']=='5' or testing_out_df.loc[i,'label']=='6' or testing_out_df.loc[i,'label']=='7' or testing_out_df.loc[i,'label']=='8' or testing_out_df.loc[i,'label']=='9'):
        testing_out_4classes_df = testing_out_4classes_df.drop([i])
        testing_in_4classes_df = testing_in_4classes_df.drop([i])

# 8 (ALL) CLASSES
training_in_8classes_df = training_in_df
training_out_8classes_df = training_out_df
testing_in_8classes_df = testing_in_df
testing_out_8classes_df = testing_out_df
for i in training_out_df.index:
    if (training_out_df.loc[i,'label']=='9'):
        training_out_8classes_df = training_out_8classes_df.drop([i])
        training_in_8classes_df = training_in_8classes_df.drop([i])
for i in testing_out_df.index:
    if (testing_out_df.loc[i,'label']=='9'):
        testing_out_8classes_df = testing_out_8classes_df.drop([i])
        testing_in_8classes_df = testing_in_8classes_df.drop([i])

# Create csv and pickle files
# 3 classes
testing_in_3classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_in_3.csv")
testing_in_3classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_in_3.pkl","wb")
pickle.dump(testing_in_3classes_df, testing_in_3classes_file)
testing_in_3classes_file.close()
testing_out_3classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_out_3.csv")
testing_out_3classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_out_3.pkl","wb")
pickle.dump(testing_out_3classes_df, testing_out_3classes_file)
testing_out_3classes_file.close()
training_in_3classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_in_3.csv")
training_in_3classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_in_3.pkl","wb")
pickle.dump(training_in_3classes_df, training_in_3classes_file)
training_in_3classes_file.close()
training_out_3classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_out_3.csv")
training_out_3classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_out_3.pkl","wb")
pickle.dump(training_out_3classes_df, training_out_3classes_file)
training_out_3classes_file.close()

# 4 classes
testing_in_4classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_in_4.csv")
testing_in_4classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_in_4.pkl","wb")
pickle.dump(testing_in_4classes_df, testing_in_4classes_file)
testing_in_4classes_file.close()
testing_out_4classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_out_4.csv")
testing_out_4classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_out_4.pkl","wb")
pickle.dump(testing_out_4classes_df, testing_out_4classes_file)
testing_out_4classes_file.close()
training_in_4classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_in_4.csv")
training_in_4classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_in_4.pkl","wb")
pickle.dump(training_in_4classes_df, training_in_4classes_file)
training_in_4classes_file.close()
training_out_4classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_out_4.csv")
training_out_4classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_out_4.pkl","wb")
pickle.dump(training_out_4classes_df, training_out_4classes_file)
training_out_4classes_file.close()

# 8 classes
testing_in_8classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_in_8.csv")
testing_in_8classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_in_8.pkl","wb")
pickle.dump(testing_in_8classes_df, testing_in_8classes_file)
testing_in_8classes_file.close()
testing_out_8classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_out_8.csv")
testing_out_8classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\testing_out_8.pkl","wb")
pickle.dump(testing_out_8classes_df, testing_out_8classes_file)
testing_out_8classes_file.close()
training_in_8classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_in_8.csv")
training_in_8classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_in_8.pkl","wb")
pickle.dump(training_in_8classes_df, training_in_8classes_file)
training_in_8classes_file.close()
training_out_8classes_df.to_csv(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_out_8.csv")
training_out_8classes_file = open(r"C:\Users\victoria.alkin\Documents\Machine Learning 2\Sets\training_out_8.pkl","wb")
pickle.dump(training_out_8classes_df, training_out_8classes_file)
training_out_8classes_file.close()