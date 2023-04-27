pip install concrete-autoencoder
pip install tensorflow
import pandas as pd
import numpy as np
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LeakyReLU

#Read in the data (whole sample)
expr = pd.read_csv("expr.csv", header = 0, index_col = 0)
expr = expr.T
expr.shape

#Read in altered data
expr = pd.read_csv("expr_no_infect_uninvolved.csv", header = 0, index_col = 0)
expr = expr.T
expr.shape

#Set up the autoencoer
x_train_expr = expr.iloc[range(90)]
x_test_expr = expr.iloc[range(90, 122)] #Change last number to match number of samples

x_train_expr = np.reshape(x_train_expr, (len(x_train_expr), -1))
x_test_expr = np.reshape(x_test_expr, (len(x_test_expr), -1))
x_train_expr.shape
x_test_expr.shape

def decoder2(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(54675)(x)
    return x
    
#Two features
selector2 = ConcreteAutoencoderFeatureSelector(K = 2, output_function = decoder2, num_epochs = 800)
selector2.fit(x_train_expr, x_train_expr, x_test_expr, x_test_expr)
selector2.get_support(indices = True)

#Four features
selector4 = ConcreteAutoencoderFeatureSelector(K = 4, output_function = decoder2, num_epochs = 800)
selector4.fit(x_train_expr, x_train_expr, x_test_expr, x_test_expr)
selector4.get_support(indices = True)

#Six features
selector6 = ConcreteAutoencoderFeatureSelector(K = 6, output_function = decoder2, num_epochs = 800)
selector6.fit(x_train_expr, x_train_expr, x_test_expr, x_test_expr)
selector6.get_support(indices = True)

#Ten features
selector10 = ConcreteAutoencoderFeatureSelector(K = 10, output_function = decoder2, num_epochs = 800)
selector10.fit(x_train_expr, x_train_expr, x_test_expr, x_test_expr)
selector10.get_support(indices = True)

#Fifteen features
selector15 = ConcreteAutoencoderFeatureSelector(K = 15, output_function = decoder2, num_epochs = 800)
selector15.fit(x_train_expr, x_train_expr, x_test_expr, x_test_expr)
selector15.get_support(indices = True)

#Twenty features
selector20 = ConcreteAutoencoderFeatureSelector(K = 20, output_function = decoder2, num_epochs = 800)
selector20.fit(x_train_expr, x_train_expr, x_test_expr, x_test_expr)
selector20.get_support(indices = True)
