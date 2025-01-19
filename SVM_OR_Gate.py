from sklearn import svm

#data points for OR gate
X = [
    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1]
]

#labels
y = [0, 1, 1, 1]

#fit the model
clf = svm.SVC(kernel='poly', degree=2)
clf.fit(X, y)

#predict
print(clf.predict([[0, 0], [1, 1], [1, 0], [0, 1]]))

#svm graph
import numpy as np
import matplotlib.pyplot as plt

# Create a mesh grid
x_min, x_max = -2, 2
y_min, y_max = -2, 2
h = .02 # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter([0, 1], [0, 1], color='black', s=80)
plt.scatter([1, 0], [0, 1], color='red', s=80)
plt.show()