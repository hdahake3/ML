import numpy as np

xCor = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
yCor = np.array([0, 1, 3, 2, 6, 4, 14, 4, 8])

xCor = [[1, x] for x in xCor]
xCorTranspose = np.transpose(xCor)

result = np.linalg.inv(xCorTranspose.dot(xCor)).dot(xCorTranspose.dot(yCor))

print(xCor)
print(yCor)
print(result)