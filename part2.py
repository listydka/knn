from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from vis import plot_data
X, y = [], []
with open("dataset.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        s, c, cl = parts
        X.append([int(s), int(c)])
        y.append(cl)
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
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    correct = sum(model.predict([X_test[j]])[0] == y_test[j] for j in range(len(X_test)))
    acc += correct / len(X_test)
print(f"Точность sklearn k-NN: {acc/5:.2f}")
end = datetime.now()
model.fit(X, y)
s = int(input("Сладость: "))
c = int(input("Хруст: "))
print("Класс:", model.predict([[s, c]])[0])
plot_data(X, y, [s, c])
print("Время работы программы:", end - start)
