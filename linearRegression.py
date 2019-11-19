import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

def linearRegression():

    d = load_diabetes() #load data
    
    d_X = d.data.shape[:,np.newaxis,2] #feature array shape is 2 dimensional
    
    #the last 20 observations for testing and use the rest for training your model.
    #seperate train and test data
    dx_train = d_X[:-20]
    dy_train = d.target[:-20] #target vector shape - 1 dimensional

    dx_test = d_X[-20:]
    dy_test = d.target[-20:]

    #set up linear regression model
    model = LinearRegression()

    #use fit
    model.fit(dx_train, dy_train)

    #calculate the mean square error
    mse = np.mean(model.predict(dx_test - dy_test) **2)
    
    #check the score
    model_score = model.score(dx_test, dy_test)

    #After fitting the model, it can be applied
    print(model.coef_)
    print(mse)
    print(model_score)

    '''cd = {dx_train:'r', dy_train:'r'}
    cd2 = {dx_test:'g', dy_test:'g'}
    cols = np.array((cd(target) for target in target))'''

    plt.scatter(dx_train, dy_train)
    plt.plot(dx_train, model.predict(dx_train), c='b')
    plt.show()

    return()

print(linearRegression())






