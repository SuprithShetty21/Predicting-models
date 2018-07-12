import csv 
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []



#def date_fixer(strg):
#    strg1 = strg.split('-')
#    strg1 = ''.join(strg1)
#    return strg1



def get_data(filename):
    with open(filename,'r') as f:
        reader_=csv.reader(f)
        next(reader_)
        next(reader_)
        for row in reader_:
            #dates.append(int(date_fixer(row[0])))
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return



def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel='linear', C=1e3)          #1e3 = 1000
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', gamma=0.1)

    svr_lin = svr_lin.fit(dates, prices)
    svr_poly = svr_poly.fit(dates, prices)
    svr_rbf = svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='data')
    plt.plot(dates, svr_lin.predict(dates), color='red', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.plot(dates, svr_rbf.predict(dates), color='green', label='RBF model')
    plt.xlabel('Dates')
    plt.ylabel('prices')
    plt.title('Support Vector Regression models')
    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]

get_data('stocks.csv')
print(dates)
print(prices)
predictedprice = predict_prices(dates, prices, 290117)
print('Your model is ready: ')