import os
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import math



def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    assert len(points[0].values) == len(points[1].values)
    assert len(points[0].values) % 20 == 0
    return points[0].values, points[1].values

def view_data_segments(xs, ys):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

def fit_wh(X,Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y) 

def sum_squared_error(y,y_hat):
    return np.sum((y- y_hat) ** 2)

def least_squares_linear(x, y):
    x_e = poly_fill(x,1)
    v = fit_wh(x_e,y) 
    y_hat = v[0] + (v[1]  * x)
    return y_hat,sum_squared_error(y,y_hat)

def cubic_least_squares(x, y):
    x_e = poly_fill(x,3)
    v = fit_wh(x_e,y)
    y_hat = np.power(x,3) * v[3] + (x**2) * v[2] + x * v[1] + v[0]
    return y_hat,sum_squared_error(y,y_hat)

def sin_least_squares(x, y):
    x_e = sin_fill(x)
    v = fit_wh(x_e,y)
    y_hat = v[0] + (v[1]  * np.sin(x))
    return y_hat,sum_squared_error(y,y_hat)

def poly_fill(x, degree):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    for i in range(2, degree + 1):
        x_e = np.column_stack((x_e, x**i))
    return(x_e)

def sin_fill(x):
    return np.column_stack((np.ones(x.shape),np.sin(x)))


#This is just random.shuffle but implement it here so q is set to 0.6 always gives us the correct results 
def my_shuffle(x):
    p = x
    for i in reversed(range(1,len(x))):
        q = random.random()
        j = math.floor(q * (i + 1))
        p[i],p[j] = p[j],p[i]
    return p

# creates k arrays of the same length keeping x y pairings and shuffles them
def k_folds(xs,ys,k):
    temp = list(zip(xs,ys))
    temp = my_shuffle(temp)
    nxs,nys = zip(*temp)
    return np.array_split(np.array(nxs),k),np.array_split(np.array(nys),k)

def remove_this_element(arr,idx):
    j = (arr[[idx]])
    n = j[-1]
    x = (np.nonzero(np.all(n==arr,axis=1)))
    return (np.delete(arr,x,axis=0)),arr[x][0]


#returns the mean cv error
def validation(xs,ys,k):
    folds_xs,folds_ys = k_folds(xs,ys,k)
    lin_error = []
    sin_error = []
    pol_error = []
    folds_xs = np.array(folds_xs)
    folds_ys = np.array(folds_ys)
  
    for i in range(k):
        X_train_set,X_test = (remove_this_element(folds_xs,i))
        Y_train_set,Y_test = (remove_this_element(folds_ys,i))

        X_train_sets =  np.concatenate(X_train_set)
        Y_train_sets =  np.concatenate(Y_train_set)


        train_lin = fit_wh(poly_fill(X_train_sets,1),Y_train_sets)
        train_poly = fit_wh(poly_fill(X_train_sets,3),Y_train_sets)
        train_sin = fit_wh(sin_fill(X_train_sets),Y_train_sets)


        Yh_lin = poly_fill(X_test,1).dot(train_lin)
        Yh_pol = poly_fill(X_test,3).dot(train_poly)
        Yh_sin = sin_fill(X_test).dot(train_sin)



        sin_error.append(sum_squared_error(Y_test,Yh_sin))
        lin_error.append(sum_squared_error(Y_test,Yh_lin))
        pol_error.append(sum_squared_error(Y_test,Yh_pol)) 

    return np.mean(lin_error),np.mean(sin_error),np.mean(pol_error)

def min_y_hat(lin_y_hat,lin_error,l_mean,cub_y_hat,cub_error,p_mean,sin_y_hat,sin_error,s_mean):
    min_error = min(s_mean,l_mean,p_mean)
    percentdiff = 0
    if min_error < l_mean:
        percentdiff = ((l_mean - min_error) / min_error)  * 100
    if percentdiff > 7:
        if(min_error == l_mean):
            # print("Linear")
            return lin_y_hat,lin_error
        elif(min_error == s_mean):
            # print("Unknown")
            return sin_y_hat,sin_error
        else:
            # print("Polynomial")
            return cub_y_hat, cub_error
    else:
        #  print('Linear')
         return lin_y_hat,lin_error

def best_y_hat(x,y,k):
    lin_cv,sin_cv ,pol_cv = [],[],[]
    for i in range(k): 
        lin_cv1,sin_cv1,pol_cv1 = validation(x,y,20)
        lin_cv.append(lin_cv1)
        sin_cv.append(sin_cv1)
        pol_cv.append(pol_cv1)
    
    lin_cv = np.array(lin_cv)
    sin_cv = np.array(sin_cv)
    pol_cv = np.array(pol_cv)


    linear_fit,linear_error = least_squares_linear(x,y)
    cubic_fit, cubic_error = cubic_least_squares(x,y)
    sin_fit,sin_error = sin_least_squares(x,y)
    # print("--------------------------------------------------------------")

    # print("Linear CV {0} and Linear ssm {1}".format(lin_cv.mean(),linear_error))
    # print("cubic  CV {0} and cubic  ssm {1}".format(pol_cv.mean(),cubic_error))
    # print("sin    CV {0} and sin    ssm {1}".format(sin_cv.mean(),sin_error))
    # print("-------------------------------------------------------------- \n")



    y_hat,error = min_y_hat(linear_fit,linear_error,lin_cv.mean(),cubic_fit,
    cubic_error,pol_cv.mean(),sin_fit,sin_error,sin_cv.mean())
    return y_hat,error

def get_best_fit(xs,ys):
    final = []
    errors = []
    for x,y in zip(xs,ys):
        #if using random Set value change 1 to 50 or more
        y_hat,error = best_y_hat(x,y,20)
        final.extend(y_hat)
        errors.append(error)
    return final,errors


def plot(weights,path):
    xs,ys = load_points_from_file(path)
    plt.plot(xs,weights)
    view_data_segments(xs,ys)
    plt.show


def get_segments(xs,ys):
    return np.split(xs,len(xs)/20),np.split(ys,len(xs)/20)

def seg_help(path):
    xs,ys = load_points_from_file(path)
    return get_segments(xs,ys)

def print_error(path,plots=False):
    try:
        xs,ys = seg_help(path)
        xs = np.array(xs)
        ys= np.array(ys)
        best_y_hat,error = get_best_fit(xs,ys)
        print(sum(error))
        if plots:
            plot(best_y_hat,path)
        return(sum(error))
    except(FileNotFoundError,IOError):
        print('FileNotFound')
    
    

def run_all():
    directory = 'datafiles/train_data'
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".csv") :
                file = (os.path.join(directory, filename))
                print(file)
                print_error(file)
            else:
                continue
    except(FileNotFoundError):
        print("Directory not found")


def main():
    if (len(sys.argv) >= 3 and sys.argv[2] != '--plot'):
        print('Incorrect Inputs')
    elif len(sys.argv) == 1:
        run_all()
    elif len(sys.argv) == 2:
        print_error(sys.argv[1])
    elif sys.argv[2] == '--plot':
        print_error(sys.argv[1],plots=True)






if __name__== "__main__":
    
    main()


