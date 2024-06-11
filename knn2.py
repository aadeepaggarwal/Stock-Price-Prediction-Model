import csv
import random
import math
import operator
import datetime

import matplotlib.pyplot as plt

def loadDataset(filename, split, trainingSet=[], testSet=[], content_header=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(1, len(content_header) - 1):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(1, length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance) - 1

    for x in range(len(trainingSet)):
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

def getData(filename, stockname, startdate, enddate):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        # Assuming the CSV file has a header row
        header = dataset[0]
        # Get the index of relevant columns
        date_index = header.index('Date')
        open_index = header.index('Open')
        high_index = header.index('High')
        low_index = header.index('Low')
        close_index = header.index('Close')  # Assuming this is yesterday's closing
        # Iterate through the data
        for row in dataset[1:]:
            # Extract relevant information
            date_str = row[date_index]
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            # Check if the date is within the desired range
            if startdate <= date <= enddate:
                open_price = float(row[open_index])
                high_price = float(row[high_index])
                low_price = float(row[low_index])
                close_price = float(row[close_index])
                # Calculate state change
                state_change = change(close_price, float(row[close_index-1]))  # Assuming close_price is adjusted close
                # Append the formatted data
                formatted_row = [date_str, open_price, high_price, low_price, close_price, state_change]
                dataset.append(formatted_row)
    return dataset


def change(today, yest):
    if today > yest:
        return 'up'
    return 'down'

def predictFor(k, filename, stockname, startdate, enddate, writeAgain, split):
    iv = ["date", "open", "high", "low", "yesterday closing adj", "state change"]
    trainingSet = []
    testSet = []

    if writeAgain:
        print("Loading data from", filename)
        getData(filename, stockname, startdate, enddate)

    loadDataset(filename, split, trainingSet, testSet, iv)

    print("Predicting for", stockname)
    print("Train:", len(trainingSet))
    print("Test:", len(testSet))
    totalCount = len(trainingSet) + len(testSet)
    print("Total:", totalCount)

    predict_and_get_accuracy(testSet, trainingSet, k, stockname)

def predict_and_get_accuracy(testSet, trainingSet, k, stockname):
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)

    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:', accuracy, '%')

    plt.figure(2)
    plt.title("Prediction vs Actual Trend of " + stockname)
    plt.legend(loc="best")
    row = []
    col = []
    for dates in range(len(testSet)):
        new_date = datetime.datetime.strptime(testSet[dates][0], "%Y-%m-%d")
        row.append(new_date)
        if predictions[dates] == "down":
            col.append(-1)
        else:
            col.append(1)
    predicted_plt, = plt.plot(row, col, 'r', label="Predicted Trend")

    row = []
    col = []
    for dates in range(len(testSet)):
        new_date = datetime.datetime.strptime(testSet[dates][0], "%Y-%m-%d")
        row.append(new_date)
        if testSet[dates][-1] == "down":
            col.append(-1)
        else:
            col.append(1)
    actual_plt, = plt.plot(row, col, 'b', label="Actual Trend")

    plt.legend(handles=[predicted_plt, actual_plt])
    plt.show()

def main():
    split = 0.67
    startdate = datetime.datetime(2002, 1, 1)
    enddate = datetime.datetime.now()
    predictFor(5, 'data/amtd.csv', 'AMTD', startdate, enddate, True, split)
main()
