import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv("regression - Sheet1.csv")
print(df)
#%matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='green',marker='+')
reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
print(reg.predict([[3300]]))
