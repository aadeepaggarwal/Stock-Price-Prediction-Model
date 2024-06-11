import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y, progress_callback=None): #main knn function
        self.X_train = X
        self.y_train = y
        if progress_callback is not None:
            for _ in range(len(X)):
                progress_callback()

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x): #main output function /evaluation function
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = [self.y_train[i] for i in nearest_indices]
        most_common = max(set(nearest_labels), key=nearest_labels.count)
        return most_common 

# Split data into train and test sets
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    return X_train.values, X_test.values, y_train.values, y_test.values

def accuracy_score_custom(y_true, y_pred):
    return np.mean(y_true == y_pred)

def run(data):
    # Split features and target variable
    X = data[['Open', 'High', 'Low', 'Close']]
    y = data['Flag']

    # Normalize features
    X_normalized = (X - X.min()) / (X.max() - X.min())

    X_train, X_test, y_train, y_test = train_test_split_custom(X_normalized, y, test_size=0.2, random_state=42)

    # Instantiate the custom KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Fit the model with progress tracking
    with tqdm(total=len(X_train), desc="Training") as pbar:
        knn.fit(X_train, y_train, progress_callback=lambda: pbar.update(1))

    # Make predictions
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score_custom(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Visualize actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), y_test, label='Actual', color='blue')
    plt.plot(np.arange(len(y_test)), y_pred, label='Predicted', color='red')
    plt.title('Actual vs. Predicted')
    plt.xlabel('Index')
    plt.ylabel('Flag')
    plt.legend()
    plt.show()
    plt.savefig('Plots/amazon.png')
    plt.close()

def main():
    data_files = [file for file in os.listdir('data') if file.endswith('.csv')]
    for file in data_files:
        file_path = os.path.join('data', file)
        data = pd.read_csv(file_path)
        run(data)

if __name__ == "__main__":
    main()
