# Packages

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Import dataset

data = pd.read_csv('fitness_poses_csvs_out_basic.csv', header=None)

# Prepare data

data.columns = [str(col) for col in data.columns]
data.drop(['0'], axis=1)

def newtarget(row):
    if row['1'] == 'deadlift_down':
        return 0
    elif row['1'] == 'deadlift_up':
        return 1

data['1'] = data.apply(newtarget, axis=1)

X = data.drop(['0','1'], axis=1)
y = data['1']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=1)

# Initialize the model and set the hyperparameters
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Train
knn.fit(X_train, y_train)


## Score -- DEBUGGING PURPOSES
#print("Test data accuracy was", knn.score(X_test, y_test))
#print("Train data accuracy was", knn.score(X_train, y_train))

#Save the model
pkl_file = 'repcounter.p'
with open(pkl_file, 'wb') as file:
    pickle.dump(knn, file)