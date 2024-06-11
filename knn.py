import pandas as pd
import os
import numpy as np
import math
import operator
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(1, length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance) - 1
    for x in range((len(trainingSet))):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distance.append((trainingSet[x], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def getAccuracy1(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if RMSD(testSet[x][-1], predictions[x]) < 1:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def RMSD(X, Y):
    return math.sqrt(pow(Y - X, 2))

def change(today, yest):
    if today > yest:
        return 'up'
    return 'down'

def load_stock_data(directory):
    """
    Load stock data from files in the specified directory.

    Args:
    - directory (str): Path to the directory containing the stock data files.

    Returns:
    - dict: Dictionary containing company names as keys and DataFrame containing
            the stock data as values.
    """
    stock_data = {}

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            company_name = filename.split('.')[0]
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            stock_data[company_name] = df

    return stock_data

def predict_and_get_accuracy(testSet, trainingSet, k, stockname):
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    plt.figure(2)
    plt.title("Prediction vs Actual Trend of " + stockname)
    plt.legend(loc="best")
    row = []
    col = []
    for dates in range(len(testSet)):
        new_date = datetime.datetime.strptime(testSet[dates][0], "%Y-%M-%d")
        row.append(new_date)
        if predictions[dates]== "down":
            col.append(-1)
        else:
            col.append(1)
    predicted_plt, = plt.plot(row, col, 'r', label="Predicted Trend")
    row = []
    col = []
    for dates in range(len(testSet)):
        new_date = datetime.datetime.strptime(testSet[dates][0], "%Y-%M-%d")
        row.append(new_date)
        if testSet[dates][-1]== "down":
            col.append(-1)
        else:
            col.append(1)
    actual_plt, = plt.plot(row, col, 'b', label="Actual Trend")
    plt.legend(handles=[predicted_plt, actual_plt])
    plt.show()

# Example usage:
directory_path = 'data'
stock_data = load_stock_data(directory_path)

for file in os.listdir(directory_path):
    if file.endswith('.csv'):
        company_name = file.split('.')[0]
        print(f"Loaded data for {company_name}.")

