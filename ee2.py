from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from vis import plot_data

X,y=[],[]
for line in open("dataset.csv",encoding="utf-8"):
    s,c,cl=line.strip().split(",")
    X.append([int(s),int(c)]); y.append(cl)

start = datetime.now()

n=len(X); fold=n//5; acc=0
for i in range(5):
    test=list(range(i*fold,(i+1)*fold))
    train=[j for j in range(n) if j not in test]

    model=KNeighborsClassifier(n_neighbors=3)
    model.fit([X[j] for j in train],[y[j] for j in train])

    correct=sum(model.predict([X[j]])[0]==y[j] for j in test)
    acc+=correct/len(test)

print("Точность:",round(acc/5,2))
print("Время алгоритма:", datetime.now()-start)

model.fit(X,y)

s=int(input("Сладость: "))
c=int(input("Хруст: "))

start2 = datetime.now()
res = model.predict([[s,c]])[0]
end2 = datetime.now()

print("Класс:", res)
print("Время классификации:", end2 - start2)

plot_data(X,y,[s,c])
