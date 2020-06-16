import numpy as np
import matplotlib.pyplot as plotter
import random as rand

xCor = [rand.randint(0,400)/100 for x in range(9)] + [rand.randint(600,1000)/100 for x in range(9)]
yCor = [rand.randint(0,400)/100 for y in range(9)] + [rand.randint(600,1000)/100 for y in range(9)]
zCor = [0 for x in range(9)] + [1 for x in range(9)]

colormap = np.array(['r', 'b'])
print(xCor)
print(yCor)
print(zCor)

plotter.scatter(xCor, yCor, c=colormap[zCor])



def predOut(v0, v1, v2, x1, x2):
    x2 = v0 + v1 * x1 + v2 * x2
    return 1 / (1 + np.exp(-x2))

def costFunc(xVal, yVal, zVal, v0, v1, v2):
    sum = 0
    for i in range(len(xVal)):
        predVal = predOut(v0, v1, v2, xVal[i], yVal[i])
        y = zVal[i]
        sum += -y * np.log(predVal + 1e-300) - (1 - y) * np.log(1 - predVal + 1e-300)
    return (1 / len(xVal)) * sum

def partDeriv(partVal, xVal, yVal, zVal, v0, v1, v2, isv0):
    sum = 0
    for i in range(len(partVal)):
        predVal = predOut(v0, v1, v2, xVal[i], yVal[i])
        sum += (predVal - zVal[i]) * partVal[i] if not isv0 else predVal - zVal[i]

    return sum * (1 / len(partVal))

def gradientDesc(xVal, yVal, zVal, learningRate, runTimes):
    v0 = 0
    v1 = 0
    v2 = 0
    x = np.arange(-20, 20)
    for i in range(runTimes):
        print(costFunc(xVal, yVal, zVal, v0, v1, v2), v0, v1, v2)
        temp0 = v0 - learningRate * partDeriv(xVal, xVal, yVal, zVal, v0, v1, v2, True)
        temp1 = v1 - learningRate * partDeriv(xVal, xVal, yVal, zVal, v0, v1, v2, False)
        temp2 = v2 - learningRate * partDeriv(yVal, xVal, yVal, zVal, v0, v1, v2, False)
        v0 = temp0
        v1 = temp1
        v2 = temp2
    plotter.plot(x, -(v1/v2)*x-(v0/v2))

    return [v0, v1, v2]

gradientDesc(xCor, yCor, zCor, 1, 100000)

plotter.show()