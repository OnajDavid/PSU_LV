import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import confusion_matrix
from sklearn import tree
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
# ucitaj podatke za ucenje
df = pd.read_csv('C:\\Users\\student\\Downloads\\LV5-20250402\\occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']
plt.figure()
plt.title('Tree of CO2')
X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,stratify=y,random_state=43)
scaler = StandardScaler()
scaler = scaler.fit(X)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
tree.plot_tree(clf)
plt.show()