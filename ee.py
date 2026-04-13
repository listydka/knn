from collections import Counter
import math
from datetime import datetime
from vis import plot_data
X,y=[],[]
for line in open("dataset.csv",encoding="utf-8"):
    s,c,cl=line.strip().split(",")
    X.append([float(s),float(c)]); y.append(cl)

start = datetime.now()

n=len(X); fold=n//5; acc=0
for i in range(5):
    test=range(i*fold,(i+1)*fold)
    train=[j for j in range(n) if j not in test]
    correct=sum(
        Counter(y[t] for t in sorted(train, key=lambda t:math.dist(X[j],X[t]))[:3]).most_common(1)[0][0]==y[j]
        for j in test)
    acc+=correct/fold

print("Точность:",round(acc/5,2))
print("Время алгоритма:", datetime.now()-start)

def knn(x,k=3):
    d=[math.dist(x,p) for p in X]
    idx=sorted(range(len(d)), key=lambda i:d[i])[:k]
    return Counter(y[i] for i in idx).most_common(1)[0][0]
s=float(input("Сладость: "))
c=float(input("Хруст: "))

start2 = datetime.now()
res = knn([s,c])
end2 = datetime.now()

print("Класс:", res)
print("Время классификации:", end2 - start2)

plot_data(X,y,[s,c])
