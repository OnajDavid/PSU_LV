import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("C:\\Users\\student\\Desktop\\lv2\\mtcars.csv", "rb"), usecols = (1,2,3,4,5,6), delimiter = ',', skiprows = 1)
max = 0.0
min = 100.0
count = 0
sum = 0
for mpg in data[:,0]:
    if max < mpg:
        max = mpg
    if min > mpg:
        min = mpg
    count += 1
    sum += mpg
avg = sum/count
print(max,min,avg)

max = 0.0
min = 100.0
count = 0
counter = 0
sum = 0
for cyl in data[:,1]:
    if cyl == 6:
        mpg = data[count,0]
        if max < mpg:
            max = mpg
        if min > mpg:
            min = mpg
        sum += mpg
        counter += 1
    count += 1
avg = sum/counter
print(max,min,avg)
plt.scatter(data[:,0],data[:,4], marker='.',s=pow(data[:,4],4))
plt.show()