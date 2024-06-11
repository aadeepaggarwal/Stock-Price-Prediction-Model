import matplotlib.pyplot as plt   # Import matplotlib
import numpy as np
import pandas as pd
import os
import csv
import datetime
from tqdm import tqdm

for file in tqdm(os.listdir('data')):
    if file.endswith('.csv'):
        file_path = os.path.join('data', file)
        df = pd.read_csv(file_path)  # Read the CSV file
        print(df.head())  # Print the first 5 rows of the dataframe
        plt.plot(df['Date'], df['Open'])  # Plot the 'Open' column
        plt.xlabel('Date')
        plt.xticks(np.arange(0, len(df['Date']), 365), rotation=45)
        plt.ylabel('Open')
        plt.title(file)
        plt.savefig(f'Plots/{file[:-4]}.png')
        plt.show()