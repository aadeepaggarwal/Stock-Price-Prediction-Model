import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
import seaborn as sns 
import os 
from datetime import datetime 

import warnings 
warnings.filterwarnings("ignore") 

data = pd.read_csv('./data/amazon.csv') 
print(data.shape) 
print(data.sample(7)) 

data.info() 

data['date'] = pd.to_datetime(data['date']) 
data.info() 

data['date'] = pd.to_datetime(data['date']) 
# date vs open 
# date vs close 
plt.figure(figsize=(15, 8)) 
for index, company in enumerate(companies, 1): 
	plt.subplot(3, 3, index) 
	c = data[data['Name'] == company] 
	plt.plot(c['date'], c['close'], c="r", label="close", marker="+") 
	plt.plot(c['date'], c['open'], c="g", label="open", marker="^") 
	plt.title(company) 
	plt.legend() 
	plt.tight_layout() 

plt.figure(figsize=(15, 8)) 
for index, company in enumerate(companies, 1): 
	plt.subplot(3, 3, index) 
	c = data[data['Name'] == company] 
	plt.plot(c['date'], c['volume'], c='purple', marker='*') 
	plt.title(f"{company} Volume") 
	plt.tight_layout() 

