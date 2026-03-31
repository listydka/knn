from collections import Counter
import math
from datetime import datetime
from vis import plot_data
start = datetime.now()
X, y = [], []
with open("ds.txt", encoding="utf-8") as f:
    for line in f:
        s, c, cl = line.strip().split(",")
        X.append([float(s), float(c)])
        y.append(cl)
def knn(x,X_train, y_train, k=3):
    d = [math.sqrt((x[0]-p[0])**2 + (x[1]-p[1])**2) for p in X]
    k_labels = [y[i] for i in sorted(range(len(d)), key=lambda i: d[i])[:k]]
    return Counter(k_labels).most_common(1)[0][0]

n = len(X)
fold = n // 5
acc = 0
start = datetime.now()
for i in range(5):
    test_idx = list(range(i*fold, (i+1)*fold))
    train_idx = [j for j in range(n) if j not in test_idx]
    X_train = [X[j] for j in train_idx]
    y_train = [y[j] for j in train_idx]
    X_test = [X[j] for j in test_idx]
    y_test = [y[j] for j in test_idx]
    correct = sum(knn(x, X_train, y_train) == y_test[j] for j, x in enumerate(X_test))
    acc += correct / len(X_test)
print(f"Точность k-NN: {acc/5:.2f}")
end = datetime.now()
s, c = float(input("Сладость: ")), float(input("Хруст: "))
print("Класс:", knn([s, c], X, y))
plot_data(X, y, [s,c])
print("Время работы программы:", end - start)
