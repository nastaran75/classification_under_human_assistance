import sys
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from random import seed, shuffle


class generate_data:
    def __init__(self, n, dim, list_of_std, std_y=None):
        self.n = n
        self.dim = dim
        self.list_of_std = list_of_std
        self.std_y = std_y

    def c_gaussian_linear_generate_data(self):
        self.Y = np.ones(self.n)
        minus = np.random.randint(0,self.n,self.n/2)
        # print minus
        self.Y[minus] = -1
        print self.Y
        self.X = np.zeros((self.n, self.dim))
        for idx, label in enumerate(self.Y):
            if label == +1:
                self.X[idx] = np.random.multivariate_normal([2,2],[[5,1],[1,5]])
            else:
                self.X[idx] = np.random.multivariate_normal([-2, -2], [[8, 1], [1, 3]])

        plt.scatter(self.X[:,0],self.X[:,1],c=self.Y)
        plt.show()

    def c_gaussian_kernel_generate_data(self):
        def gen_gaussian(mean_in, cov_in, class_label):
            nv = multivariate_normal(mean=mean_in, cov=cov_in)
            X = nv.rvs(self.n)
            y = np.ones(self.n, dtype=float) * class_label
            return nv, X, y

        # We will generate one gaussian cluster for each class
        mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
        mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
        mu3, sigma3 = [4, -4], [[4, 4], [2, 5]]
        mu4, sigma4 = [-4, 6], [[6, 2], [2, 3]]
        beta = np.random.binomial(1,0.5)
        nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1)  # positive class
        nv2, X2, y2 = gen_gaussian(mu2, sigma2, 1)
        nv3, X3, y3 = gen_gaussian(mu2, sigma2, -1)  # negative class

        # join the posisitve and negative class clusters
        self.X = np.vstack((X1, X2))
        self.Y = np.hstack((y1, y2))

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

        # shuffle the data
        perm = range(0, self.n * 2)
        shuffle(perm)
        self.X = self.X[perm]
        self.Y = self.Y[perm]

    # def c_gaussian_kernel_generate_data(self):
    #     # self.Y = np.ones(self.n)
    #     # minus = np.random.randint(0,self.n,self.n/2)
    #     # # print minus
    #     # self.Y[minus] = -1
    #     # print self.Y
    #     self.Y = np.zeros(self.n)
    #     self.X = np.zeros((self.n, self.dim))
    #
    #     # print beta
    #     for idx, label in enumerate(self.Y):
    #         if(random.random()>0.5):
    #             self.Y[idx] = 1
    #         else:
    #             self.Y[idx] = -1
    #         beta = np.random.randint(1,5)
    #         print beta
    #         if self.Y[idx] == +1:
    #             self.X[idx] = (beta==1 or beta==2)*np.random.multivariate_normal([2,2],[[5,1],[1,5]]) + (beta==3 or beta==4)*np.random.multivariate_normal([-2, -2], [[10, 1], [1, 3]])
    #         else:
    #             self.X[idx] = (beta==1 or beta==2)*np.random.multivariate_normal([4, -4], [[4, 4], [2, 5]]) +\
    #                           (beta==3 or beta==4)*np.random.multivariate_normal([-4, 6], [[10, 1], [1, 3]])
    #                           # (beta==3)*np.random.multivariate_normal([-4, -4], [[6, 2], [2, 3]]) +\
    #                           # (beta==4)*np.random.multivariate_normal([6, 6], [[6, 2], [8, 10]])
    #
    #     plt.scatter(self.X[:,0],self.X[:,1],c=self.Y)
    #     plt.show()




    def generate_X(self, start, end):
        self.X = rand.uniform(start, end, (self.n, self.dim))

    def c_generate_X(self):
        self.X = rand.uniform(-10, 10, (self.n, self.dim))

    def c_hard_generate_X(self):
        plus = (self.n) / 2
        minus = self.n - plus
        X_1 = rand.uniform(0.5, 10, (plus, 1))
        X_1 = np.concatenate((X_1, rand.uniform(-10, -0.5, (minus, 1))))

        X_2 = rand.uniform(-10, 10, (self.n, 1))
        # X_2 = np.concatenate((X_2, rand.uniform(-10, -0.1, (minus, 1))))

        self.X = np.concatenate((X_1, X_2), axis=1)

    def c_hard_with_offset_generate_X(self):
        plus = (self.n) / 2
        minus = self.n - plus
        X_1 = rand.uniform(3, 10, (plus, 1))
        X_1 = np.concatenate((X_1, rand.uniform(-10, 2, (minus, 1))))

        X_2 = rand.uniform(-10, 10, (self.n, 1))
        # X_2 = np.concatenate((X_2, rand.uniform(-10, -0.1, (minus, 1))))

        self.X = np.concatenate((X_1, X_2), axis=1)

    def c_kernel_generate_X(self,rad):
        frac = (self.n)/3
        self.X = rand.uniform(-10, 10, (self.n - frac, self.dim))
        print self.X.shape
        # print rand.uniform(-rad,rad,(80,2)).shape
        self.X = np.concatenate((self.X, rand.uniform(-rad, rad, (frac, self.dim))))

    def c_kernel_with_offset_generate_X(self):
        self.X = rand.uniform(-10, 10, (self.n - 60, self.dim))
        rad = 5
        print self.X.shape
        # print rand.uniform(-rad,rad,(80,2)).shape
        self.X = np.concatenate((self.X, rand.uniform(-2, 8, (60, 2))))



    def white_Gauss(self, std=0.5):
        return rand.normal(0, std, self.n)

    def sigmoid(self, x):
        return 1 / float(1 + np.exp(-x))

    def generate_Y_sigmoid(self):
        self.Y = np.array([self.sigmoid(x.sum() / float(x.shape[0])) for x in self.X])

    def generate_Y_Gauss(self):
        def gauss(x):
            divide_wt = np.sqrt(2 * np.pi) * std
            return np.exp(-(x * x) / float(2 * std * std)) / divide_wt

        std = self.std_y
        x_vec = np.array([x.sum() / float(x.shape[0]) for x in self.X])
        self.Y = np.array(map(gauss, x_vec))

    def c_generate_Y(self):
        self.Y = np.zeros((self.X.shape[0]))
        print self.Y.shape
        for idx, x in enumerate(self.X):
            if (x[0] > 0):
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
        print self.Y.shape
        # # for second figure
        for idx, x in enumerate(self.X):
            if x[0] > -2 and x[0] < 4 and random.random() < 0.5:
                self.Y[idx] = 1
            elif x[0] > -2 and x[0] < 4:
                self.Y[idx] = -1

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

    # def c_linear_with_offset_generate_Y(self):
    #     self.Y = np.zeros((self.X.shape[0]))
    #     print self.Y.shape
    #     for idx, x in enumerate(self.X):
    #         if (x[0]+x[1] > 6):
    #             self.Y[idx] = 1
    #         else:
    #             self.Y[idx] = -1
    #     print self.Y.shape
    #     # for second figure
    #     for idx, x in enumerate(self.X):
    #         if x[0]+x[1] > 4 and x[0] + x[1] <6 and random.random() < 0.5:
    #             self.Y[idx] = 1
    #         elif x[0]+x[1] > 6 and x[0]+x[1]<8 and random.random()<0.5:
    #             self.Y[idx] = -1
    #         if x[0] + x[1]>9 and random.random()<0.2:
    #             self.Y[idx] = -1
    #         elif x[0] + x[1]<0 and random.random()<0.2:
    #             self.Y[idx] = 1
    #
    #
    #     plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
    #     plt.show()

    def c_linear_with_offset_generate_Y(self):
        self.Y = np.zeros((self.X.shape[0]))
        print self.Y.shape
        for idx, x in enumerate(self.X):
            if (x[0] > 4):
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
        print self.Y.shape
        # for second figure
        for idx, x in enumerate(self.X):
            if x[0] > 2 and x[0] < 6 and random.random() < 0.5:
                self.Y[idx] = 1
            elif x[0] > 2 and x[0] < 6:
                self.Y[idx] = -1
            if x[0] > 6 and random.random() < 0.2:
                self.Y[idx] = -1
            elif x[0] < 2 and random.random() < 0.1:
                self.Y[idx] = 1

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

    def c_quan_linear_with_offset_generate_Y(self):
        self.Y = np.zeros((self.X.shape[0]))
        print self.Y.shape
        for idx, x in enumerate(self.X):
            if (np.sum(x) > 4):
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
        print self.Y.shape
        # for second figure
        for idx, x in enumerate(self.X):
            dimsum = np.sum(x)
            if dimsum > -16 and dimsum < 24 and random.random() < 0.5:
                self.Y[idx] = 1
            elif dimsum > -16 and dimsum < 24:
                self.Y[idx] = -1
            if dimsum > -16 and random.random() < 0.1:
                self.Y[idx] = -1
            elif dimsum < 24 and random.random() < 0.2:
                self.Y[idx] = 1

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

    def c_hard_generate_Y(self):
        self.Y = np.zeros((self.X.shape[0]))
        print self.Y.shape
        for idx, x in enumerate(self.X):
            if (x[0] > 0):
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
        print self.Y.shape
        # # # for second figure
        # for idx,x in enumerate(self.X):
        #     if x[0]>-1 and x[0]<1 and random.random()<0.5:
        #         self.Y[idx] = 1
        #     elif x[0]>-1 and x[0]<1:
        #         self.Y[idx] = -1

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

    def c_hard_with_offset_generate_Y(self):
        self.Y = np.zeros((self.X.shape[0]))
        print self.Y.shape
        for idx, x in enumerate(self.X):
            if (x[0] > 3):
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
        print self.Y.shape
        # # # for second figure
        # for idx,x in enumerate(self.X):
        #     if x[0]>-1 and x[0]<1 and random.random()<0.5:
        #         self.Y[idx] = 1
        #     elif x[0]>-1 and x[0]<1:
        #         self.Y[idx] = -1

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

    def c_kernel_generate_Y(self):
        self.Y = np.zeros((self.X.shape[0]))
        print self.Y.shape
        cntplus = 0
        cntminus = 0
        rad = 5
        for idx, x in enumerate(self.X):
            if (x[0] > -rad and x[0] < rad and x[1] > -rad and x[1] < rad):
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
        # minusrad = 2
        for idx, x in enumerate(self.X):
            if (x[0] > -rad and x[0] < rad and x[1] > -rad and x[1] < rad and random.random() < 0.3):
                self.Y[idx] = -1
            # else:
            #     self.Y[idx] = -1
        rnd = np.random.randint(0, self.n, (70))
        self.Y[rnd] = 1
        # print rnd
        for label in self.Y:
            if label > 0:
                cntplus += 1
            if label < 0:
                cntminus += 1
        print cntplus, cntminus
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

    def c_quan_kernel_generate_Y(self,rad,noise_ratio):
        self.Y = np.zeros((self.X.shape[0]))
        print self.Y.shape
        cntplus = 0
        cntminus = 0
        # rad = 5
        for idx, x in enumerate(self.X):
            sumdim = np.sum([dim ** 2 for dim in x])
            if sumdim<rad**2:
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
            if sumdim<(rad+2.5)**2 and sumdim>(rad-2.5)**2 and random.random()<0.5:
                self.Y[idx] = -1
            elif sumdim<(rad+2.5)**2 and sumdim>(rad-2.5)**2:
                self.Y[idx] = 1

        from mpl_toolkits import mplot3d
        ax = plt.axes(projection="3d")
        ax.scatter(self.X[:,0], self.X[:,1], self.X[:,2], c=self.Y)
        plt.show()
        # for idx, x in enumerate(self.X):
        #     flag = True
        #     for dim in x:
        #
        #         if not (dim > -rad and dim < rad ):
        #             flag=False
        #     if flag:
        #         if random.random()<noise_ratio:   # bara with offset 0.8
        #             self.Y[idx] = 1
        #         else:
        #             self.Y[idx] = -1
        #     else:
        #         if random.random()<noise_ratio:   # bara with offset 0.8
        #             self.Y[idx] = -1
        #         else:
        #             self.Y[idx] = 1

        # for idx, x in enumerate(self.X):
        #     cnt1=0
        #     cnt2=0
        #     for dim in x:
        #         if (dim > -rad-1 and dim < -rad+1 and dim>rad-1 and dim<rad+1):
        #             cnt1 += 1
        #         if (dim > -rad and dim < rad):
        #             cnt2 += 1
        #
        #     if cnt1==1 and cnt2==self.dim-1:
        #         if random.random()<0.5:   # bara with offset 0.8
        #             self.Y[idx] = 1
        #         else:
        #             self.Y[idx] = -1
            # else:
            #     if random.random()<noise_ratio:   # bara with offset 0.8
            #         self.Y[idx] = -1
            #     else:
            #         self.Y[idx] = 1

        # for idx, x in enumerate(self.X):
        #     flag = True
        #     for dim in x:
        #         if not (dim > -rad/2 and dim < rad/2):
        #             flag = False
        #     if flag:
        #         if random.random() < 1:  # bara with offset 0.8
        #             self.Y[idx] = -1
        #         else:
        #             self.Y[idx] = +1

        # for idx, x in enumerate(self.X):
        #     flag = False
        #     for dim in x:
        #         if (dim > 9 or dim<-9):
        #             flag = True
        #     if flag:
        #         self.Y[idx] =1

        # for idx, x in enumerate(self.X):
        #     flag = True
        #     for dim in x:
        #         if not (dim > -rad-1 and dim < rad+1):
        #             flag = False
        #     if flag:
        #         if random.random() < 0.5:  # bara with offset 0.8
        #             self.Y[idx] = 1
        #         else:
        #             self.Y[idx] = -1
        #
            # flag=True
            # for dim in x:
            #     if  not (dim>7 and dim<10):
            #         flag = False
            # if flag:
            #     self.Y[idx] = 1
            #
            # flag = True
            # for dim in x:
            #     if not (dim > -10 and dim < -7):
            #         flag = False
            # if flag:
            #     self.Y[idx] = 1
            #
            # # flag = True
            # # if not (dim>6 and dim<9):
            # #     flag = False
            # # if flag:
            # #     self.Y[idx] = 1
            #
            # # flag = True
            # # if not (x[0] > -9 and x[0] < -7 and x[1] < 9 and x[1] > 7):
            # #     flag = False
            # # if flag:
            # #     self.Y[idx] = 1
            #
            # # flag = True
            # # if not (x[0] > 6 and x[0] < 8 and x[1] < -6 and x[1] > -8):
            # #     flag = False
            # # if flag:
            # #     self.Y[idx] = 1
            #
            # # flag = True
            # # if not (x[0] > -9 and x[0] < -6 and x[1] < 2 and x[1] > -2):
            # #     flag = False
            # # if flag:
            # #     self.Y[idx] = 1
            #
            # flag = True
            # for dim in x:
            #     if not (dim> 0 and dim< 3):
            #         flag = False
            # if flag:
            #     self.Y[idx] = -1
            #
            #
            # # flag = True
            # # if not (x[0] > -4 and x[0] < -1 and x[1] < 3 and x[1] > 1):
            # #     flag = False
            # # if flag:
            # #     self.Y[idx] = -1
            #
            # flag = True
            # for dim in x:
            #     if not (dim> -4 and dim < -1):
            #         flag = False
            # if flag:
            #     self.Y[idx] = -1
            #
            # # flag = True
            # # if not (x[0] > 4 and x[0] < 5 and x[1] > -5 and x[1] < -4):
            # #     flag = False
            # # if flag:
            # #     self.Y[idx] = -1
            #
            # # flag = True
            # # for dim in x:
            # #     if not (dim> -5 and dim < -4 ):
            # #         flag = False
            # #     if flag:
            # #         self.Y[idx] = -1
            #
            # # flag = True
            # # if not (x[0] > -5 and x[0] < -4 and x[1] > 4 and x[1] < 5):
            # #     flag = False
            # # if flag:
            # #     self.Y[idx] = -1
            #
            # flag = True
            # for dim in x:
            #     if not (dim > -6 and dim < -5):
            #         flag = False
            #     if flag:
            #         self.Y[idx] = -1
            #
            # flag = True
            # for dim in x:
            #     if not (dim> 5 and dim < 6):
            #         flag = False
            #     if flag:
            #         self.Y[idx] = -1
            #


        # minusrad = 2
        # for idx, x in enumerate(self.X):
        #     if (x[0] > -rad and x[0] < rad and x[1] > -rad and x[1] < rad and random.random() < 0.3):
        #         self.Y[idx] = -1
        #     # else:
        #         self.Y[idx] = -1
        # rnd = np.random.randint(0, self.n, (70))
        # self.Y[rnd] = 1
        # print rnd
        for label in self.Y:
            if label > 0:
                cntplus += 1
            if label < 0:
                cntminus += 1
        print cntplus, cntminus
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

    def c_kernel_with_offset_generate_Y(self):
        self.Y = np.zeros((self.X.shape[0]))
        print self.Y.shape
        for idx, x in enumerate(self.X):
            if (x[0] > 0):
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
        print self.Y.shape
        # # # for second figure
        # for idx,x in enumerate(self.X):
        #     if(x[0]>-2 and x[0]<0):
        #         self.Y[idx] = 1
        #     elif(x[0]<2 and x[0]>0):
        #         self.Y[idx] = -1
        # for idx,x in enumerate(self.X):
        #     if(x[0]>-3 and x[0]<0 and random.random()<0.3):
        #         self.Y[idx] = -1
        #     elif (x[0] < 3 and x[0] > 0 and random.random()<0.3):
        #         self.Y[idx] = 1

        # for third figure
        cntplus = 0
        cntminus = 0
        rad = 5
        for idx, x in enumerate(self.X):
            if (x[0] > -2 and x[0] < 8 and x[1] > -2 and x[1] < 8):
                self.Y[idx] = 1
            else:
                self.Y[idx] = -1
        # minusrad = 2
        for idx, x in enumerate(self.X):
            if (x[0] > -2 and x[0] < 8 and x[1] > -2 and x[1] < 8 and random.random() < 0.3):
                self.Y[idx] = -1
            # else:
            #     self.Y[idx] = -1
        # rnd = np.random.randint(0,self.n,(70))
        # self.Y[rnd] = 1
        # print rnd
        for label in self.Y:
            if label > 0:
                cntplus += 1
            if label < 0:
                cntminus += 1
        print cntplus, cntminus
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

    def generate_Y_Mix_of_Gauss(self, no_Gauss, prob_Gauss):
        self.Y = np.zeros(self.n)
        for itr, p in zip(range(no_Gauss), prob_Gauss):
            w = rand.uniform(0, 1, self.dim)
            self.Y += p * self.X.dot(w)

    def generate_variable_human_prediction(self):
        self.c = {}
        self.human_pred = {}
        for std in self.list_of_std:
            h = np.zeros(self.Y.shape)
            self.human_pred[str(std)] = np.ones(shape=(self.Y.shape))
            for idx,label in enumerate(self.Y):
                if label == 1:
                    h[idx] = np.random.uniform(-4,8)
                if label == -1:
                    h[idx] = np.random.uniform(-8, 4)
                if h[idx]*self.Y[idx]<0:
                    self.human_pred[str(std)][idx] = -self.Y[idx]
                else:
                    self.human_pred[str(std)][idx] = self.Y[idx]
            self.c[str(std)] = np.maximum(0,1 - (self.Y * h))
            print self.c
            print self.human_pred
            # self.c[str(std)] = np.zeros(self.X.shape[0])
            # for idx, c_err in enumerate(self.c[str(std)]):
            #     if random.random()<0.5:
            #         self.c[str(std)][idx] = 1
            # self.c[str(std)] = self.variable_std_Gauss_inc(np.min(np.array(self.list_of_std)), std,
            #                                                self.X.flatten()) ** 2
            # self.c[str(std)] = np.array([0 for x in self.X])

    def variable_std_Gauss_inc(self, low, high, x):
        m = (high - low) / np.max(x)
        return np.array([rand.normal(0, m * np.absolute(x_i) + low, 1)[0] for x_i in x])

    def generate_human_prediction(self):
        self.human_pred = {}
        self.c = {}
        for std in self.list_of_std:
            h = np.zeros(self.Y.shape)
            self.human_pred[str(std)] = np.ones(shape=(self.Y.shape))
            for idx, label in enumerate(self.Y):
                if label == 1:
                    h[idx] = np.random.uniform(-2, 12)
                if label == -1:
                    h[idx] = np.random.uniform(-12, 2)
                # print h[idx]*self.Y[idx]
                if h[idx]*self.Y[idx]<0:
                    self.human_pred[str(std)][idx] = -self.Y[idx]
                else:
                    self.human_pred[str(std)][idx] = self.Y[idx]
            self.c[str(std)] = np.maximum(0, 1 - (self.Y * h))
            # print self.c[str(std)]

            # self.human_pred[str(std)] = self.Y + self.white_Gauss(std=std)
            # self.c[str(std)] = np.random.uniform(-8, 2, size=self.n)
            # self.human_pred[str(std)] = np.ones(shape=(self.c.shape))
            # for idx, hp in enumerate(self.human_pred[str(std)]):
            #     if self.c[str(std)][idx] > 0:
            #         self.human_pred[str(std)][idx] = - self.Y[idx]
            #     else:
            #         self.human_pred[str(std)][idx] = self.Y[idx]
            # self.c[str(std)] = np.maximum(0, self.c[str(std)])

    def append_X(self):
        self.X = np.concatenate((self.X, np.ones((self.n, 1))), axis=1)

    def split_data(self, frac):
        indices = np.arange(self.n)
        random.shuffle(indices)
        num_train = int(frac * self.n)
        indices_train = indices[:num_train]
        indices_test = indices[num_train:]   ##################### Doros Konnnnn
        # indices_test = indices_train
        self.Xtest = self.X[indices_test]
        self.Xtrain = self.X[indices_train]
        self.Ytrain = self.Y[indices_train]
        self.Ytest = self.Y[indices_test]
        self.human_pred_train = {}
        self.human_pred_test = {}
        self.c_train = {}
        self.c_test = {}

        for std in self.list_of_std:
            self.human_pred_train[str(std)] = self.human_pred[str(std)][indices_train]

            self.human_pred_test[str(std)] = self.human_pred[str(std)][indices_test]
            print self.human_pred_train[str(std)]
            self.c_train[str(std)] = self.c[str(std)][indices_train]
            self.c_test[str(std)] = self.c[str(std)][indices_test]

        n_test = self.Xtest.shape[0]
        n_train = self.Xtrain.shape[0]
        self.dist_mat = np.zeros((n_test, n_train))

        for te in range(n_test):
            for tr in range(n_train):
                self.dist_mat[te, tr] = LA.norm(self.Xtest[te] - self.Xtrain[tr])

        plt.subplot(1,2,1)
        plt.scatter(self.Xtrain[:,0],self.Xtrain[:,1],c=self.Ytrain)
        plt.subplot(1,2,2)
        plt.scatter(self.Xtest[:,0], self.Xtest[:,1], c=self.Ytest)
        plt.show()

        print n_test,n_train

    def visualize_data(self):
        x = self.X[:, 0].flatten()
        y = self.Y
        plt.scatter(x, y)
        plt.show()


def convert(input_data, output_data):
    def get_err(label, pred):
        # print label.shape
        # print pred.shape
        # print np.array((label!=pred),dtype=int)
        return np.array((label != pred), dtype=int) / 2
        # return (label - pred) ** 2

    data = load_data(input_data, 'ifexists')
    # print data.keys()
    list_of_std_str = data.human_pred_train.keys()
    test = {'X': data.Xtest, 'Y': data.Ytest, 'c': {}, 'y_h': {}}
    data_dict = {'test': test, 'X': data.Xtrain, 'Y': data.Ytrain, 'c': {}, 'y_h': {}, 'dist_mat': data.dist_mat}

    for std in list_of_std_str:
        data_dict['c'][std] = data.c_train[std]
        data_dict['test']['c'][std] = data.c_test[std]
        data_dict['y_h'][std] = data.human_pred_train[std]
        data_dict['test']['y_h'][std] = data.human_pred_test[std]
        print data_dict['c'][std]
        print data_dict['test']['y_h'][std]

        # data_dict['c'][std] = get_err(data_dict['Y'], data.human_pred_train[std])
        # data_dict['test']['c'][std] = get_err(data_dict['test']['Y'], data.human_pred_test[std])

    save(data_dict, output_data)


def main():
    n = 500
    dim = 2
    frac = 0.8
    list_of_options = ['gauss', 'sigmoid', 'Usigmoid', 'Ugauss', 'Wgauss', 'Wsigmoid', 'kernel',
                       'hard_linear', 'linear_with_offset', 'hard_linear_with_offset', 'kernel_with_offset',
                       'quan_linear','quan_kernel']
    options = sys.argv[1:]

    if not os.path.exists('data'):
        os.mkdir('data')

    for option in options:
        assert option in list_of_options

        input_data_file = 'data/' + option
        if option in ['Wgauss', 'Wsigmoid', 'hard_linear','hard_linear_with_offset',
                      'kernel_with_offset']:
            input_data_file = 'data/data_dict_' + option

        if option == 'sigmoid':
            list_of_std = np.array([0.001, 0.005, .01, .05])
            obj = generate_data(n, dim, list_of_std)
            obj.generate_X(-7, 7)
            obj.generate_Y_sigmoid()
            obj.generate_human_prediction()
            obj.append_X()  # for the bias
            obj.split_data(frac)
            # obj.visualize_data()
            save(obj, input_data_file)
            del obj

        if option == 'gauss':
            std_y = 2
            list_of_std = np.array([0.001, .005, 0.01, 0.05])
            obj = generate_data(n, dim, list_of_std, std_y)
            obj.generate_X(-7, 7)
            obj.generate_Y_Gauss()
            obj.generate_human_prediction()
            obj.append_X()
            obj.split_data(frac)
            save(obj, input_data_file)
            del obj

        if option == 'Usigmoid':
            list_of_std = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
            obj = generate_data(n, dim, list_of_std)
            obj.generate_X(-7, 7)
            obj.generate_Y_sigmoid()
            obj.generate_human_prediction()
            obj.append_X()
            obj.split_data(frac)
            # obj.visualize_data()
            save(obj, input_data_file)
            del obj

        if option == 'Ugauss':
            std_y = 2
            list_of_std = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
            obj = generate_data(n, dim, list_of_std, std_y)
            obj.generate_X(-7, 7)
            obj.generate_Y_Gauss()
            obj.generate_human_prediction()
            obj.append_X()
            obj.split_data(frac)
            save(obj, input_data_file)
            del obj

        if option == 'Wsigmoid':
            list_of_std = [0.001]
            obj = generate_data(n=240, dim=1, list_of_std=list_of_std)
            obj.generate_X(-7, 7)
            obj.generate_Y_sigmoid()
            obj.generate_variable_human_prediction()
            obj.append_X()
            full_data = {}
            for std in list_of_std:
                full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
            save(full_data, input_data_file)

        if option == 'Wgauss':
            list_of_std = [0.001]
            obj = generate_data(n=240, dim=1, list_of_std=list_of_std, std_y=2)
            obj.generate_X(-1, 1)
            obj.generate_Y_Gauss()
            obj.generate_variable_human_prediction()
            obj.append_X()
            full_data = {}
            for std in list_of_std:
                full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
            save(full_data, input_data_file)

        if option == 'linear':
            list_of_std = [0.001]
            obj = generate_data(n=240, dim=2, list_of_std=list_of_std, std_y=2)
            obj.c_generate_X()
            obj.c_generate_Y()
            obj.generate_variable_human_prediction()
            # obj.append_X()
            full_data = {}
            for std in list_of_std:
                full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
            print obj.X.shape
            print obj.Y.shape
            save(full_data, input_data_file)

        if option == 'linear_with_offset':  # bara hame soft ha hamino dadam
            list_of_std = [0.001]
            obj = generate_data(n=300, dim=2, list_of_std=list_of_std, std_y=2)
            obj.c_gaussian_linear_generate_data()

            # obj.c_generate_X()
            # obj.c_linear_with_offset_generate_Y()
            obj.generate_human_prediction()
            obj.split_data(frac)
            # obj.append_X()
            # full_data = {}
            # for std in list_of_std:
            #     full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
            print obj.X.shape
            print obj.Y.shape
            save(obj, input_data_file)

        if option == 'hard_linear':
            list_of_std = [0.001]
            obj = generate_data(n=240, dim=2, list_of_std=list_of_std, std_y=2)
            obj.c_hard_generate_X()
            obj.c_hard_generate_Y()
            obj.generate_variable_human_prediction()
            # obj.append_X()
            full_data = {}
            for std in list_of_std:
                full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
            print obj.X.shape
            print obj.Y.shape
            save(full_data, input_data_file)

        if option == 'hard_linear_with_offset':
            list_of_std = [0.001]
            obj = generate_data(n=240, dim=2, list_of_std=list_of_std, std_y=2)
            obj.c_hard_with_offset_generate_X()
            obj.c_hard_with_offset_generate_Y()
            obj.generate_variable_human_prediction()
            # obj.append_X()
            full_data = {}
            for std in list_of_std:
                full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
            print obj.X.shape
            print obj.Y.shape
            save(full_data, input_data_file)

        if option == 'kernel':
            list_of_std = [0.001]
            obj = generate_data(n=300, dim=2, list_of_std=list_of_std, std_y=2)
            obj.c_kernel_generate_X(5)
            obj.c_quan_kernel_generate_Y(5,0.9)
            # obj.generate_variable_human_prediction()
            obj.generate_human_prediction()
            obj.split_data(frac)
            # obj.append_X()
            # full_data = {}
            # for std in list_of_std:
            #     full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)],'y_h':obj.human_pred[str(std)]}
            print obj.X.shape
            print obj.Y.shape
            save(obj, input_data_file)

        # if option == 'kernel_with_offset':   #estefade nakardam
        #     list_of_std = [0.001]
        #     obj = generate_data(n=240, dim=2, list_of_std=list_of_std, std_y=2)
        #     obj.c_kernel_with_offset_generate_X()
        #     obj.c_kernel_with_offset_generate_Y()
        #     obj.generate_variable_human_prediction()
        #     # obj.append_X()
        #     full_data = {}
        #     for std in list_of_std:
        #         full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
        #     print obj.X.shape
        #     print obj.Y.shape
        #     save(full_data, input_data_file)

        if option == 'quan_linear':
            list_of_std = np.array([1.0])
            obj = generate_data(n, dim, list_of_std)
            obj.generate_X(-10, 10)
            obj.c_quan_linear_with_offset_generate_Y()
            obj.generate_human_prediction()
            obj.split_data(frac)
            print obj.X.shape
            print obj.Y.shape
            # obj.visualize_data()
            save(obj, input_data_file)
            del obj

        if option == 'quan_kernel':
            list_of_std = ['1.0']
            obj = generate_data(n=1000, dim=2, list_of_std=list_of_std, std_y=2)
            # obj.c_kernel_generate_X(5)
            # obj.c_generate_X()
            # obj.c_quan_kernel_generate_Y(12, 1)
            obj.c_gaussian_kernel_generate_data()
            # obj.generate_variable_human_prediction()
            obj.generate_human_prediction()
            obj.split_data(frac)
            # obj.append_X()
            # full_data = {}
            # for std in list_of_std:
            #     full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)],'y_h':obj.human_pred[str(std)]}
            print obj.X.shape
            print obj.Y.shape
            save(obj, input_data_file)

        if option not in ['Wgauss', 'Wsigmoid', 'hard_linear',
                          'hard_linear_with_offset']:
            if os.path.exists('data/data_dict_' + option + '.pkl'):
                os.remove('data/data_dict_' + option + '.pkl')
            output_data_file = 'data/data_dict_' + option
            print 'converting'
            convert(input_data_file, output_data_file)


if __name__ == "__main__":
    main()
