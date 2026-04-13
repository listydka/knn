from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import random
from datetime import datetime
from ee2 import X, y

data=list(zip(X,y))
random.shuffle(data)
X,y=map(list,zip(*data))

def cv(k=3):
    n=len(X); fold=max(1,n//5); acc=0
    for i in range(5):
        test=list(range(i*fold,(i+1)*fold))
        train=[j for j in range(n) if j not in test]

        model=KNeighborsClassifier(n_neighbors=k)
        model.fit([X[j] for j in train],[y[j] for j in train])

        correct=sum(model.predict([X[j]])[0]==y[j] for j in test)
        acc+=correct/len(test)
    return acc/5

print("\nТест на количество соседей")
for k in [1,3,5,7,9]:
    print("k=",k,"Точность",round(cv(k),2))

print("\nТест на размер выборки")
for size in [10,20,40,80,len(X)]:
    Xs,ys=X[:size],y[:size]

    def cv_size():
        n=len(Xs); fold=max(1,n//5); acc=0
        for i in range(5):
            test=list(range(i*fold,(i+1)*fold))
            train=[j for j in range(n) if j not in test]

            model=KNeighborsClassifier(n_neighbors=3)
            model.fit([Xs[j] for j in train],[ys[j] for j in train])

            correct=sum(model.predict([Xs[j]])[0]==ys[j] for j in test)
            acc+=correct/len(test)
        return acc/5

    print("Размер",size,"Точность",round(cv_size(),2))

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)
pred=model.predict(X)
print("\nМатрица ошибок")
print(confusion_matrix(y,pred))
