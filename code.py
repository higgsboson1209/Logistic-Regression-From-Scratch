import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
mydataset=pd.read_csv('data.txt')
X=mydataset.iloc[:,:-1]
Y = mydataset.iloc[:, -1]
admitted = mydataset.loc[Y == 1]
notadmitted = mydataset.loc[Y == 0]
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(notadmitted.iloc[:, 0], notadmitted.iloc[:, 1], s=10, label='Not Admitted')
X = np.c_[np.ones(99),X]
print(X)
Y = np.c_[Y]
theta = np.zeros((3, 1))
def sigmoid(z):
    return 1/(1+np.exp(-z))
for i in range(0,100000):
    z=np.dot(X,theta)
    h=sigmoid(z)
    grad=np.dot(X.T,(h-Y))/Y.size
    theta=theta-(0.001)*grad
print(theta)
x_ = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
x_values=np.linspace(x_[0],x_[1])
y_values = - (theta[0] + theta[1]*x_values)/theta[2]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()
