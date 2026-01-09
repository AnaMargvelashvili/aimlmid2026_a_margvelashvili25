import csv
import random
from math import exp

# =========================
# 1. Load CSV data
# =========================

file_path = "../data/a_margvelashvili25_19582.csv"

with open(file_path, newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

print("Total rows (including header):", len(rows))
print("Header:", rows[0])

# =========================
# 2. Prepare features (X) and labels (y)
# =========================

X = []
y = []

for row in rows[1:]:
    features = [
        int(row[0]),  # words
        int(row[1]),  # links
        int(row[2]),  # capital_words
        int(row[3])   # spam_word_count
    ]
    label = int(row[4])  # is_spam

    X.append(features)
    y.append(label)

print("Total samples:", len(X))

# =========================
# 3. Split data: 70% train / 30% test
# =========================

data = list(zip(X, y))
random.shuffle(data)

split_index = int(0.7 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================
# 4. Logistic Regression (from scratch)
# =========================

def sigmoid(z):
    return 1 / (1 + exp(-z))

def predict(features, weights, bias):
    z = sum(f * w for f, w in zip(features, weights)) + bias
    return sigmoid(z)

weights = [0.0] * len(X_train[0])
bias = 0.0
learning_rate = 0.001
epochs = 1000

for _ in range(epochs):
    for features, label in zip(X_train, y_train):
        y_hat = predict(features, weights, bias)
        error = label - y_hat

        for i in range(len(weights)):
            weights[i] += learning_rate * error * features[i]
        bias += learning_rate * error

print("Model trained.")
print("Weights:", weights)
print("Bias:", bias)

# =========================
# 5. Evaluation: Confusion Matrix & Accuracy
# =========================

TP = FP = TN = FN = 0

for features, label in zip(X_test, y_test):
    prob = predict(features, weights, bias)
    prediction = 1 if prob >= 0.5 else 0

    if prediction == 1 and label == 1:
        TP += 1
    elif prediction == 1 and label == 0:
        FP += 1
    elif prediction == 0 and label == 0:
        TN += 1
    elif prediction == 0 and label == 1:
        FN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)

print("\nConfusion Matrix:")
print("TP:", TP, "FP:", FP)
print("FN:", FN, "TN:", TN)
print("Accuracy:", round(accuracy, 4))
