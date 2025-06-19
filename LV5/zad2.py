import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
# ucitaj podatke za ucenje
df = pd.read_csv('C:\\Users\\student\\Downloads\\LV5-20250402\\occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,stratify=y,random_state=43)
scaler = StandardScaler()
scaler = scaler.fit(X)
K = []
training = []
test = []
scores = {}
plt.figure(1)
for k in range(2,91):
    clf = knc(n_neighbors=k*5)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    K.append(k) 
    training.append(train_score) 
    test.append(test_score) 
    scores[k] = [train_score, test_score] 

plt.scatter(K,training,color='k')
plt.scatter(K,test,color='g')
plt.xlabel('Round #')
plt.ylabel('Accuracy')
plt.title('Preciznost po klasama')
plt.legend()
plt.show()
print(scores)


cls = knc(n_neighbors=8)
cls.fit(X_train,y_train)
y_pred = cls.predict(X_test)

matrix = confusion_matrix(y_test,y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=matrix)
display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
