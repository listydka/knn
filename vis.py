import matplotlib.pyplot as plt

def plot_data(X, y, test_point=None):
    colors = {"Фрукт":"orange","Овощ":"green","Протеин":"blue"}
    for cl in set(y):
        xs = [X[i][0] for i in range(len(X)) if y[i]==cl]
        ys = [X[i][1] for i in range(len(X)) if y[i]==cl]
        plt.scatter(xs, ys, c=colors.get(cl,'gray'), label=cl)
    if test_point:
        plt.scatter(test_point[0], test_point[1], c='red', marker='x', s=100, label="Тест")
    plt.xlabel("Сладость"); plt.ylabel("Хруст")
    plt.legend(); plt.grid(); plt.show()
