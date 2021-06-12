import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

def Confusion_Matrix(theta, x, y):
    '''
    returns

    TP       FN
                  
    FP       TN

    '''
    threshold = 0.5

    z = propagate(x, theta)
    y_pred = (z>=threshold)

    cm = np.zeros((2, 2))

    n = y.shape[0]

    for i in range(n):
        if y[i] == 1:
            if y_pred[i] == 1:
                cm[0][0] += 1
            
            elif y_pred[i] == 0:
                cm[0][1] += 1

        elif y[i] == 0:
            if y_pred[i] == 1:
                cm[1][0] += 1
            
            elif y_pred[i] == 0:
                cm[1][1] += 1

    return cm

def Metrics(cm):
    '''
    accuracy, recall, precision, f-score

    '''

    a = cm[0][0]
    b = cm[0][1]
    c = cm[1][0]
    d = cm[1][1]

    acc = (a + d)/(a+b+c+d)
    acc = 100*acc
    recall = ((a/(a+b))*(a+c) + (d/(d+c))*(b+d))/(a+b+c+d)
    precision = ((a/(a+c))*(a+b)  +  (d/(d+b))*(c+d))/(a+b+c+d)
    fscore = (((2*a/(b+c+2*a))*(a+c))  +  ((2*d/(b+c+2*d))*(b+d)))/(a+b+c+d)

    return acc, recall, precision, fscore



def propagate(x, W):
    z = x@W
    a = sigmoid(z)
    return a

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def loss(y, f):
    m = f.shape[0]
    loss = (-1/m)*(y.T@np.log(f) + (1 - y).T @ np.log(1 - f)) 

    return loss


def accuracy(x, y, theta):
    threshold = 0.5

    z = propagate(x, theta)
    y_pred = (z>=threshold)

    n = y.shape[0]

    count = 0

    for i in range(n):
        if (y_pred[i] == y[i]):
            count += 1

    acc = (count * 100)/n

    return acc


def gradient_descent(x, y, eta, iters):
    n_f = x.shape[1]
    m = y.shape[0]
    y = np.reshape(y, (m, 1))
    costs = []
    accuracies = []
    theta = np.zeros((n_f, 1))

    for i in range(iters):
        a = propagate(x, theta)
        theta = theta - (eta/m)*(x.T@(a-y))
        
        c = loss(y, a)
        if i%50 == 0:
            cost = loss(y, propagate(x, theta))
            costs.append(cost)
            accuracies.append(accuracy(x, y, theta))

    return theta, costs, accuracies

def stochastic(x, y, eta, n_iters):
    n_f = x.shape[1]
    m = y.shape[0]
    y = np.reshape(y, (m, 1))
    costs = []
    accuracies = []
    theta = np.zeros((n_f, 1))

    for i in range(n_iters):
        randint = np.random.randint(0, m)
        x_i = x[randint].reshape(n_f, 1)
        y_i = y[randint].reshape(1, 1)

        a = propagate(x_i.T, theta)
        theta = theta - (eta)*(x_i@(a - y_i))

        if i%50 == 0:
            cost = loss(y, propagate(x, theta))
            costs.append(cost)
            accuracies.append(accuracy(x, y, theta))

    return theta, costs, accuracies

def solve_plot(data):
    np.random.shuffle(data)
    n = data.shape[0]      #number of examples
    m = data.shape[1] - 1  #number of features
    train_size = int(0.7*n)
    test_size = n - train_size

    x_train = data[:train_size, 0:m]
    y_train = data[:train_size, m]

    x_test = data[train_size:n, 0:m]
    y_test = data[train_size:n, m]
    y_test = y_test.reshape(test_size, 1)

    eta = [0.1, 0.001, 0.01]
    n_iters = 10000
    colors = ['r', 'b', 'g']

    gd_accuracies = []
    gd_costs = []

    for i in range(3):
        gd_theta, gd_cost, gd_accuracy = gradient_descent(x_train, y_train, eta[i], n_iters)
        
        gd_accuracy = np.squeeze(np.array(gd_accuracy))
        gd_accuracies.append(gd_accuracy)
        
        gd_cost = np.squeeze(np.array(gd_cost))
        gd_costs.append(gd_cost)

        print("GD Weight : ", gd_theta)

    for i in range(3):
        plt.plot(gd_accuracies[i], colors[i], label = str(eta[i]))
    
    plt.title("Training Accuracy for Gradient Descent")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()
    plt.clf()

    for i in range(3):
        plt.plot(gd_costs[i], colors[i], label = str(eta[i]))

    plt.title("Training Costs for Gradient Descent")
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()
    plt.clf()   
        

    sgd_accuracies = []
    sgd_costs = []

    for i in range(3):
        sgd_theta, sgd_cost, sgd_accuracy = stochastic(x_train, y_train, eta[i], 50000)
        
        sgd_accuracy = np.squeeze(np.array(sgd_accuracy))
        sgd_accuracies.append(sgd_accuracy)
        
        sgd_cost = np.squeeze(np.array(sgd_cost))
        sgd_costs.append(sgd_cost)

        print("SGD Weights : ", sgd_theta)

    for i in range(3):
        plt.plot(sgd_accuracies[i], colors[i], label = str(eta[i]))

    plt.title("Training Accuracy for Stochastic Gradient Descent")
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()
    plt.clf()

    for i in range(3):
        plt.plot(sgd_costs[i], colors[i], label = str(eta[i]))

    plt.title("Training Costs for Stochastic Gradient Descent")
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()
    plt.clf()
    


def solve(data):
    np.random.shuffle(data)
    n = data.shape[0]      #number of examples
    m = data.shape[1] - 1  #number of features
    train_size = int(0.7*n)
    test_size = n - train_size

    x_train = data[:train_size, 0:m]
    y_train = data[:train_size, m]

    x_test = data[train_size:n, 0:m]
    y_test = data[train_size:n, m]
    y_test = y_test.reshape(test_size, 1)

    eta = 0.001 #learning rate
    n_iter = 1000

    gd_theta, gd_costs, gd_train_accu = gradient_descent(x_train, y_train, eta, n_iter)
    
    gd_costs = np.squeeze(np.array(gd_costs))
    gd_train_cost = gd_costs[-1]

    gd_train_accu = gd_train_accu[-1]

    gd_cm = Confusion_Matrix(gd_theta, x_test, y_test)
    gd_train_cm = Confusion_Matrix(gd_theta, x_train, y_train)

    f = propagate(x_test, gd_theta)
    gd_test_loss = loss(y_test, f)

    
    sgd_theta, sgd_costs, sgd_train_acc = stochastic(x_train, y_train, 0.1, 50000)
    
    sgd_costs = np.squeeze(np.array(sgd_costs))
    sgd_train_cost = sgd_costs[-1]

    sgd_train_acc = sgd_train_acc[-1]

    sgd_cm = Confusion_Matrix(sgd_theta, x_test, y_test)
    sgd_train_cm = Confusion_Matrix(sgd_theta, x_train, y_train)

    f = propagate(x_test, sgd_theta)
    sgd_test_loss = loss(y_test, f)

    return gd_test_loss, gd_train_cost, gd_cm, gd_train_cm, sgd_test_loss, sgd_train_cost, sgd_cm, sgd_train_cm
    
    


if __name__ == "__main__":
    dataset = pd.read_csv("C:/Users/kodey/Downloads/dataset_LR.csv")

    data = dataset.values
    n = data.shape[0]

    y = data[0:,-1:]
    x = data[0:,0:-1]

    #adding bias coeff to x and concatenating y back on x
    ones=np.ones((x.shape[0],1))
    x = np.concatenate((ones,x),axis=1)
    data=np.concatenate((x,y),axis=1)

    gd_test_avg_met = []
    gd_train_avg_met = []

    sgd_test_avg_met = []
    sgd_train_avg_met = []

    solve_plot(data)
    '''
    for i in range(10):
        gd_test_loss, gd_train_cost, gd_cm, gd_train_cm, sgd_test_loss, sgd_train_cost, sgd_cm, sgd_train_cm = solve(data)

        gd_test_acc, gd_test_recall, gd_test_precision, gd_test_fscore = Metrics(gd_cm)

        gd_train_acc, gd_train_recall, gd_train_precision, gd_train_fscore = Metrics(gd_train_cm)

        gd_test_avg_met.append([gd_test_loss, gd_test_acc, gd_test_recall, gd_test_precision, gd_test_fscore])
        gd_train_avg_met.append([gd_train_cost, gd_train_acc, gd_train_recall, gd_train_precision, gd_train_fscore])
        
        print("\n", "\n")

        print("Training Accuracy for GD : ", gd_train_acc)
        print("Training Loss for GD : ", gd_train_cost)
        print("Training Recall for GD : ", gd_train_recall)
        print("Training Precision for GD : ", gd_train_precision)
        print("Training F-Score for GD : ", gd_train_fscore)

        print("\n")
        print("Testing Accuracy for GD : ", gd_test_acc)
        print("Testing Loss for GD : ", gd_test_loss)
        print("Testing Recall for GD : ", gd_test_recall)
        print("Testing Precision for GD : ", gd_test_precision)
        print("Testing F-Score for GD : ", gd_test_fscore)

        #print("Confusion_Matrix for GD : ", gd_cm)
        


        sgd_test_acc, sgd_test_recall, sgd_test_precision, sgd_test_fscore = Metrics(sgd_cm)

        sgd_train_acc, sgd_train_recall, sgd_train_precision, sgd_train_fscore = Metrics(sgd_train_cm)

        sgd_test_avg_met.append([sgd_test_loss, sgd_test_acc, sgd_test_recall, sgd_test_precision, sgd_test_fscore])
        sgd_train_avg_met.append([sgd_train_cost, sgd_train_acc, sgd_train_recall, sgd_train_precision, sgd_train_fscore])
        
        print("\n", "\n")
        print("Training Accuracy for SGD : ", sgd_train_acc)
        print("Training Loss for SGD : ", sgd_train_cost)
        print("Training Recall for SGD : ", sgd_train_recall)
        print("Training Precision for SGD : ", sgd_train_precision)
        print("Training F-Score for SGD : ", sgd_train_fscore)

        print("\n")
        print("Testing Accuracy for SGD : ", sgd_test_acc)
        print("Testing Loss for SGD : ", sgd_test_loss)
        print("Testing Recall for SGD : ", sgd_test_recall)
        print("Testing Precision for SGD : ", sgd_test_precision)
        print("Testing F-Score for SGD : ", sgd_test_fscore)
        
        
        #print("Confusion_Matrix for SGD : ", sgd_cm)

    
    
    gd_test_avg_met = np.mean(np.array(gd_test_avg_met), axis = 0)
    gd_train_avg_met = np.mean(np.array(gd_train_avg_met), axis = 0)

    sgd_test_avg_met = np.mean(np.array(sgd_test_avg_met), axis = 0)
    sgd_train_avg_met = np.mean(np.array(sgd_train_avg_met), axis = 0)


    print("\n", "\n", "\n")
    

    print("Average Training Accuracy for GD : ", gd_train_avg_met[1])
    print("Average Training Loss for GD : ", gd_train_avg_met[0])
    print("Average Training Recall for GD : ", gd_train_avg_met[2])
    print("Average Training Precision for GD : ", gd_train_avg_met[3])
    print("Average Training F-Score for GD : ", gd_train_avg_met[4])
    print("\n")

    print("Average Testing Accuracy for GD : ", gd_test_avg_met[1])
    print("Average Testing Loss for GD : ", gd_test_avg_met[0])
    print("Average Testing Recall for GD : ", gd_test_avg_met[2])
    print("Average Testing Precision for GD : ", gd_test_avg_met[3])
    print("Average Testing F-Score for GD : ", gd_test_avg_met[4])


    print("\n", "\n", "\n")

    print("Average Training Accuracy for SGD : ", sgd_train_avg_met[1])
    print("Average Training Loss for SGD : ", sgd_train_avg_met[0])
    print("Average Training Recall for SGD : ", sgd_train_avg_met[2])
    print("Average Training Precision for SGD : ", sgd_train_avg_met[3])
    print("Average Training F-Score for SGD : ", sgd_train_avg_met[4])

    print("\n")

    print("Average Testing Accuracy for SGD : ", sgd_test_avg_met[1])
    print("Average Testing Loss for SGD : ", sgd_test_avg_met[0])
    print("Average Testing Recall for SGD : ", sgd_test_avg_met[2])
    print("Average Testing Precision for SGD : ", sgd_test_avg_met[3])
    print("Average Testing F-Score for SGD : ", sgd_test_avg_met[4])

'''