import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')
mtcars = mtcars.sort_values(by='mpg',ascending=False)
print(mtcars[0:5])
mtcars = mtcars.sort_values(by='mpg')
print(mtcars[0:3].query('cyl == 8'))
midcars = mtcars.query('cyl == 6')
sum = midcars.sum().mpg
sum = sum/len(midcars)
print(sum)
smallcars = mtcars.query('cyl == 4 & wt <= 2.2 & wt > 2.0')
avg = (smallcars.sum().wt)/len(smallcars)
print(avg)
print(len(mtcars.query('am == 1'))," ", len(mtcars.query('am == 0')))
print(len(mtcars.query('am == 1 & hp > 100')))
mtcars['wt'] = mtcars.wt * 1000
print(mtcars)