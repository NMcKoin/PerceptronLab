import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Perceptron class
class Perceptron:
   
    def __init__(self, n_inputs, lr=0.1, seed=12345):
        """
        Constructor:
            self.W: weights, one for each input
            self.b: bias, one for the entire perceptron
            self.lr: learning rate
        """
        np.random.seed(seed)
        self.W = np.random.randn(n_inputs)
        self.b = 0.0
        self.lr = lr

    def activation(self, z):
        """
        Simulates the binary activation
        """
        return 1 if z >= 0 else 0 

    def predict(self, x):
        """
        Applies the linear transformation, then passes the result through activation
        """
        z = np.dot(self.W, x) + self.b
        return self.activation(z)

    def train(self, X, y, epochs=10):
        """
         Iterates through the dataset for a given number of epochs, updating the weights and bias
        """
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                # Make prediction
                pred = self.predict(xi)

                # Compute prediction error
                error = yi - pred

                # Update all weights and bias
                self.W += self.lr * error * xi
                self.b += self.lr * error



iris = load_iris()
X = iris.data[:, [2, 3]]  # petal length and petal width

mask = iris.target < 2 # keep setosa and versicolor classes only

X = X[mask]
y = iris.target[mask]

print("X.shape:", X.shape)
print("Classes:", np.unique(y))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  random_state=42
  )

# Create a perceptron instance and train it
p = Perceptron(
  n_inputs=2,
  lr=0.1
  )

p.train(
    X_train,
    y_train,
    epochs=100
    )

# predict on the test set
y_pred = np.array(
    [p.predict(x) for x in X_test]
)

# saving predictions
np.savetxt(
    ".//output/perceptron_predictions.txt",
    y_pred,
    delimiter=",",
    fmt="%d"
)


accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Accuracy:", accuracy)

# Visualize the predictions
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c = y_pred,
    cmap='coolwarm',
    edgecolor='black',
    )

plt.title("Perceptron Predictions")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.savefig(".//output/perceptron_predictions.png")
plt.close()


# adjusting the learning rate and number of epochs
for lr in [0.01, 0.1, 0.5]:
    for epochs in [10, 50, 100, 500]:
        p = Perceptron(
            n_inputs=2,
            lr=lr
        )
        p.train(
            X_train,
            y_train,
            epochs=epochs
        )
        y_pred = np.array(
            [p.predict(x) for x in X_test]
        )
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Learning Rate: {lr}, Epochs: {epochs}, Accuracy: {accuracy:.3f}")