import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def norm_pdf(arr, mu, var):
    y = (1/math.sqrt(2*np.pi*var))*(np.exp(-1*((arr - mu)**2)/(2*var)))

    return y


class linear_discriminant:
    def __init__(self, data):
        self.data = data

        self.n_pts = self.data.shape[0]
        self.D = self.data.shape[1] - 1
        print(self.D)

        self.class_1 = []
        self.class_2 = []
        self.n_pts_1 = 0
        self.n_pts_2 = 0

        self.mean_1 = np.zeros(self.D)
        self.mean_2 = np.zeros(self.D)

        self.Sw = np.zeros([self.D, self.D])

        self.w = np.zeros(self.D)


    def separate(self):
        for i in range(self.n_pts):
            if self.data[i][self.D] == 0:
                self.n_pts_1 += 1
                self.class_1.append(self.data[i][0:self.D])

            else:
                self.n_pts_2 += 1
                self.class_2.append(self.data[i][0:self.D])

        self.class_1 = np.array(self.class_1)
        self.class_2 = np.array(self.class_2)
        

    def get_mean_matrices(self):  
        self.mean_1 = np.sum(self.class_1, axis=0) / self.n_pts_1

        self.mean_2 = np.sum(self.class_2, axis=0) / self.n_pts_2           


    def generate_Sw(self):
        
        temp_1 = (self.class_1 - self.mean_1).T

        temp_2 = (self.class_2 - self.mean_2).T

        self.Sw = ((temp_1@temp_1.T)/self.n_pts_1) + ((temp_2@temp_2.T)/self.n_pts_2) 

    
    def generate_w(self):
        self.w = np.linalg.inv(self.Sw)@(self.mean_1 - self.mean_2)

        self.w = self.w/math.sqrt((sum(np.square(self.w))))
        print(self.w)

    
    def generate_projections(self):
        wT = self.w.reshape((1, self.D))

        self.projections_1 = wT@self.class_1.T
        
        self.projections_2 = wT@self.class_2.T

        self.projections_1 = self.projections_1[0]
        
        self.projections_2 = self.projections_2[0]


    def generate_threshold(self):
        mu_1 = np.sum(self.projections_1)/self.n_pts_1
        mu_2 = np.sum(self.projections_2)/self.n_pts_2

        var_1 = np.var(self.projections_1)
        var_2 = np.var(self.projections_2)

        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.var_1 = var_1
        self.var_2 = var_2

        std_1 = math.sqrt(var_1)
        std_2 = math.sqrt(var_2)

        a = 1/(2*std_1**2) - (2*std_2**2)

        b = mu_2/(std_2**2) - mu_1/(std_1**2)

        c = mu_1**2/(2*std_1**2) - mu_2**2/(2*std_2**2) - np.log(std_2/std_1)

        x = np.roots([a, b, c])

        x1 = x[0]
        x2 = x[1]

        t_1 = pow((x1 - mu_1), 2) + pow((x1 - mu_2), 2)
        t_2 = pow((x2 - mu_1), 2) + pow((x2 - mu_2), 2)

        if (t_1<t_2):
            self.b = x1

        else:
            self.b = x2

        print(mu_1, var_1)
        print(mu_2, var_2)

        print(self.b)


    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.class_1[:, 0], self.class_1[:, 1], self.class_1[:, 2], s = 1, c='r')
        ax.scatter(self.class_2[:, 0], self.class_2[:, 1], self.class_2[:, 2], s = 1, c='b')

        xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
        
        z = (self.b - xx*self.w[0] - yy*self.w[1]) / self.w[2] 

        # plot the plane
        ax.plot_surface(xx, yy, z, alpha=0.5)

        plt.show()

        domain = np.linspace(-2, 2, 1000) 
        plt.clf()
        plt.plot(domain, norm_pdf(domain, self.mu_1, self.var_1))
        plt.plot(domain, norm_pdf(domain, self.mu_2, self.var_2))
        plt.plot([[self.b], [self.b]],[[-1], [2]])
        plt.show()

        plt.clf()
        zeros_1 = np.ones(self.n_pts_1)
        zeros_2 = np.ones(self.n_pts_2)

        zeros_1 = -1 * zeros_1
        zeros_2 = -1 * zeros_2

        plt.plot(self.projections_1, zeros_1, 'g^')
        plt.plot(self.projections_2, zeros_2, 'r+')
        plt.plot([[self.b], [self.b]],[[-2], [1]])
        plt.show()

    
    def print_accuracy(self):
        class_1_mis = 0
        class_1_prop = 0
        for i in range(len(self.projections_1)):
            if self.projections_1[i] > self.b:
                class_1_prop += 1

            else:
                class_1_mis += 1

        class_2_mis = 0
        class_2_prop = 0
        for i in range(len(self.projections_2)):
            if self.projections_2[i] < self.b:
                class_2_prop += 1

            else:
                class_2_mis += 1

        print("Number of properly classified points of class 1 = ", class_1_prop)
        print("Number of misclassified points of class 1 = ", class_1_mis)
        print("Number of properly classified points of class 2 = ", class_2_prop)
        print("Number of misclassified points of class 2 = ", class_2_mis)

        accuracy = 100*(class_1_prop+class_2_prop)/(class_1_prop+class_1_mis+class_2_mis+class_2_prop)
        print("Accuracy on Training Data = ", accuracy, "%")


def solve(data):
    model = linear_discriminant(data)

    model.separate()

    model.get_mean_matrices()

    model.generate_Sw()

    model.generate_w()

    model.generate_projections()

    model.generate_threshold()

    model.plot()

    model.print_accuracy()


if __name__ == "__main__":
    np.set_printoptions(precision=6)
    df = pd.read_csv("C:/Users/kodey/Documents/Python Scripts/ML - Assignment/dataset_FLD.csv", header=None)
    data = df.values
    solve(data)