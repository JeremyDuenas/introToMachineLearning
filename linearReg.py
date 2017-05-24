from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xValues = np.array([1,2,3,4,5,6], dtype=np.float64)
yValues = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope_and_intercept(xValues, yValues):
    m = ( ((mean(xValues) * mean(yValues)) - mean(xValues * yValues)) /
    ((mean(xValues) * mean(xValues)) - mean(xValues * xValues)))
    b = mean(yValues) - m * mean(xValues)
    return m, b

def squared_error (ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

m, b= best_fit_slope_and_intercept(xValues, yValues)

print(m,b)

regression_line = [(m*x) + b for x in xValues]

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(yValues, regression_line)

print(r_squared)

plt.scatter(xValues, yValues)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xValues, regression_line)
plt.show()
