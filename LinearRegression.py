import numpy as np
import matplotlib.pyplot as plotter


xCor = [0, 1, 2, 3, 4, 5, 6, 7, 8]
yCor = [0, 1, 3, 2, 6, 4, 14, 4, 8]

plotter.scatter(xCor, yCor)
plotter.plot(np.arange(0, len(xCor) + 1), np.arange(0, len(xCor) + 1) * 0)


def predOut(v0, v1, x):
    return v0 + v1 * x

def costFunc(xVal, yVal, v0, v1):
    sum = 0
    for i in range(len(xVal)):
        predVal = predOut(v0, v1, xVal[i])
        sum += (predVal - yVal[i])**2
    return (1/(2 * len(xVal))) * sum

def partDeriv(xVal, yVal, v0, v1, isv0):
    sum = 0
    for i in range(len(xVal)):
        predVal = predOut(v0, v1, xVal[i])
        sum += (predVal - yVal[i]) * xVal[i] if not isv0 else predVal - yVal[i]

    return sum * (1/len(xVal))

def gradientDesc(xVal, yVal, learningRate, runTimes):
    v0 = 0
    v1 = 0
    x = np.arange(0, len(xVal) + 1)
    for i in range(runTimes):
        temp0 = v0 - learningRate * partDeriv(xVal, yVal, v0, v1, True)
        temp1 = v1 - learningRate * partDeriv(xVal, yVal, v0, v1, False)
        v0 = temp0
        v1 = temp1
        plotter.plot(x, v0 + v1 * x)
    return [v0, v1]


print(gradientDesc(xCor, yCor, 0.01, 10000))
plotter.show()
