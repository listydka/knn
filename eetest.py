from collections import Counter
import math
import random
from datetime import datetime
from ee import X, y, knn

data = list(zip(X, y))
random.shuffle(data)
X, y = map(list, zip(*data))

def cv(Xs, ys, k=3):
    n=len(Xs); fold=max(1,n//5); acc=0
    for i in range(5):
        test=range(i*fold,min((i+1)*fold,n))
        train=[j for j in range(n) if j not in test]

        correct=sum(
            Counter(ys[t] for t in sorted(train, key=lambda t:math.dist(Xs[j],Xs[t]))[:k]).most_common(1)[0][0]==ys[j]
            for j in test)

        acc+=correct/len(list(test))
    return acc/5

print("\nТест на количество соседей")
for k in [1,3,5,7,9]:
    print("k=",k,"Точность=",round(cv(X,y,k),2))

print("\nТест на размер выборки")
for size in [10,20,40,80,len(X)]:
    Xs,ys=X[:size],y[:size]
    print("Размер",size,"Точность",round(cv(Xs,ys),2))

print("\nТест шума")
Xn=[[x[0]+random.uniform(-0.5,0.5),
     x[1]+random.uniform(-0.5,0.5)] for x in X]
print("Точность с шумом",round(cv(Xn,y),2))
