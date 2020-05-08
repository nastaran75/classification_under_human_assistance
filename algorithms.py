import math
from baseline_classes import *
import numpy as np
import numpy.linalg as LA
import random
import time


class triage_human_machine:
    def __init__(self, data_dict, real=None):
        self.X = data_dict['X']
        self.Y = data_dict['Y']

        if real:
            self.c = data_dict['c']

        else:
            self.c = np.square(data_dict['human_pred'] - data_dict['Y'])

        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)
        self.epsilon = float(1)
        self.BIG_VALUE = 100000
        self.real = real

    def get_c(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def get_minus(self, subset, elm):
        return np.array([i for i in subset if i != elm])

    def get_added(self, subset, elm):
        return np.concatenate((subset, np.array([int(elm)])), axis=0)


    def set_param(self, lamb, K):
        self.lamb = lamb
        self.K = K

    def get_optimal_pred(self, subset):
        subset_c = self.get_c(subset)
        X_sub = self.X[subset_c].T
        Y_sub = self.Y[subset_c]
        subset_c_l = self.n - subset.shape[0]
        return LA.inv(self.lamb * subset_c_l * np.eye(self.dim) + X_sub.dot(X_sub.T)).dot(X_sub.dot(Y_sub))

    def plot_subset(self, w, subset):
        plt_obj = {}

        x = self.X[subset, 0].flatten()
        y = self.Y[subset]
        plt_obj['human'] = {'x': x, 'y': y}

        c_subset = self.get_c(subset)
        x = self.X[c_subset, 0].flatten()
        y = self.Y[c_subset]
        plt_obj['machine'] = {'x': x, 'y': y}

        x = self.X[:, 0].flatten()
        y = self.X.dot(w).flatten()
        plt_obj['prediction'] = {'x': x, 'y': y, 'w': w}
        return plt_obj

    def stochastic_distort_greedy(self, g, K, gamma, epsilon,svm_type):
        print 'in stochastic'
        c_mod = modular_distort_greedy({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb,'svm_type':svm_type})
        subset = np.array([]).astype(int)
        g.reset()
        epsilon=0.001
        s = int(math.ceil(self.n * np.log(float(1) / epsilon) / float(K)))
        print 'subset_size', s, 'K-->', K, ', n --> ', self.n

        for itr in range(K):
            frac = (1 - gamma / float(K)) ** (K - itr - 1)
            # print frac
            subset_c = self.get_c(subset)
            # print subset_c.shape

            # subset_choosen = np.array(random.sample(np.arange(self.n),s))

            if s < subset_c.shape[0]:
                subset_choosen = np.array(random.sample(subset_c, s))
            else:
                subset_choosen = subset_c
            # print subset_choosen
            c_mod_inc = c_mod.get_inc_arr(subset, rest_flag=True, subset_rest=subset_choosen)
            g_inc_arr, subset_c_ret = g.get_inc_arr(subset, rest_flag=True, subset_rest=subset_choosen)
            g_pos_inc = g_inc_arr.flatten() + c_mod_inc
            inc_vec = frac * g_pos_inc - c_mod_inc

            if np.max(inc_vec) <= 0:
                print 'no increment'
                # continue
                return subset

            sel_ind = np.argmax(inc_vec)
            elm = subset_choosen[sel_ind]
            subset = self.get_added(subset, elm)
            g.update_data_str(elm)

        return subset

    def distort_greedy(self, g, K, gamma,svm_type):
        c_mod = modular_distort_greedy({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb,'svm_type':svm_type})
        subset = np.array([]).astype(int)
        g.reset()
        for itr in range(K):
            frac = (1 - gamma / float(K)) ** (K - itr - 1)

            subset_c = self.get_c(subset)
            c_mod_inc = c_mod.get_inc_arr(subset).flatten()
            g_inc_arr, subset_c_ret = g.get_inc_arr(subset)
            g_pos_inc = g_inc_arr.flatten() #+ c_mod_inc
            inc_vec = frac * g_pos_inc - c_mod_inc

            if np.max(inc_vec) <= 0:
                print 'no increment'
                # continue
                return subset

            sel_ind = np.argmax(inc_vec)
            elm = subset_c[sel_ind]
            subset = self.get_added(subset, elm)
            g.update_data_str(elm)
        return subset

    def gamma_sweep_distort_greedy(self, svm_type, delta=0.01, T=5, flag_stochastic=None):

        T=1
        print T
        g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb, 'svm_type':svm_type})
        Submod_ratio = 0.7
        # if T != 5:
        #     delta = 0.05
        subset = {}
        G_subset = []
        # gamma = 1.0
        gamma = 0.1
        # gamma = float(1) / float(self.n)

        for r in range(T + 1):
            if flag_stochastic:
                subset_sel = self.stochastic_distort_greedy(g, self.K, gamma, delta,svm_type=svm_type)
            else:
                subset_sel = self.distort_greedy(g, self.K, gamma, svm_type=svm_type)
            subset[str(r)] = subset_sel
            G_subset.append(g.eval(subset_sel))
            gamma = gamma * (1 - delta)
        empty_set = np.array([]).astype(int)
        subset[str(T + 1)] = empty_set
        G_subset.append(g.eval(empty_set))
        max_set_ind = np.argmax(np.array(G_subset))

        return subset[str(max_set_ind)]

    def kl_triage_subset(self):
        kl_obj = kl_triage({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        return kl_obj.get_subset(self.K)


    def algorithmic_triage(self, param, optim):
        # start=time.time()
        self.set_param(param['lamb'], int(param['K'] * self.n))

        # if optim == 'RLSR':
        #     subset = self.CRR_subset()
        #
        # if optim == 'RLSR_Reg':
        #     subset = self.CRR_Reg()
        #
        # if optim == 'diff_submod':
        #     subset = self.sel_subset_diff_submod()
        #
        # if optim == 'greedy':
        #     subset = self.max_submod_greedy()
        #
        # if optim == 'diff_submod_greedy':
        #     subset = self.sel_subset_diff_submod_greedy()
        # start = time.time()
        if optim == 'distort_greedy':
            # subset = self.gamma_sweep_distort_greedy(T=param['DG_T'], flag_stochastic=False)
            subset = self.gamma_sweep_distort_greedy(svm_type=param['svm_type'])
        # finish = time.time()
        # print finish - start

        if optim == 'kl_triage':
            subset = self.kl_triage_subset()
        #
        if optim == 'stochastic_distort_greedy':
            subset = self.gamma_sweep_distort_greedy(flag_stochastic=True, svm_type=param['svm_type'])

        if subset.shape[0] == self.n:
            w_m = 0
        else:
            w_m = self.get_optimal_pred(subset)
        # print w_m

        plt_obj = {'w': w_m, 'subset': subset}
        return plt_obj
