import numpy as np
import matplotlib.pyplot as plotter

xCor = [0, 1, 2, 3, 9, 10, 11, 12, 13]
yCor = [0, 0, 0, 0, 1, 1, 1, 1, 1]

plotter.scatter(xCor, yCor)



def predOut(v0, v1, x):
    x2 = v0 + v1 * x
    return 1 / (1 + np.exp(-x2))

def costFunc(xVal, yVal, v0, v1):
    sum = 0
    for i in range(len(xVal)):
        predVal = predOut(v0, v1, xVal[i])
        y = yVal[i]
        sum += -y * np.log(predVal + 1e-300) - (1 - y) * np.log(1 - predVal + 1e-300)
    return (1 / len(xVal)) * sum

def partDeriv(xVal, yVal, v0, v1, isv0):
    sum = 0
    for i in range(len(xVal)):
        predVal = predOut(v0, v1, xVal[i])
        sum += (predVal - yVal[i]) * xVal[i] if not isv0 else predVal - yVal[i]

    return sum * (1 / len(xVal))

def gradientDesc(xVal, yVal, learningRate, runTimes):
    v0 = 0
    v1 = 0
    x = np.arange(-20, 20)
    for i in range(runTimes):
        print(costFunc(xVal, yVal, v0, v1), v0, v1)
        temp0 = v0 - learningRate * partDeriv(xVal, yVal, v0, v1, True)
        temp1 = v1 - learningRate * partDeriv(xVal, yVal, v0, v1, False)
        v0 = temp0
        v1 = temp1
        plotter.plot(x, predOut(v0, v1, x))
        
    return [v0, v1]

print(gradientDesc(xCor, yCor, 1, 100000))

plotter.show()