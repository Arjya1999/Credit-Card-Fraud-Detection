#EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv('creditcard.csv')
dataset.head()

#Checking if any columns contain null values
dataset.isnull().sum()

dataset.info()

dataset.describe()

#Plotting the correlation map
plt.figure(figsize=(20,10))
sns.heatmap(dataset.corr(),annot=True)

#Correlation with respect to independent variable
dataset.corrwith(dataset.Class).plot.bar(figsize=(20,10),title='Correlation with independent variable',fontsize=15,rot=45,grid=True)
