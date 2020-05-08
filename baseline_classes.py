from myutil import *

class kl_triage:
    def __init__(self, data):
        self.X = data['X']
        self.Y = data['Y']
        self.c = data['c']
        self.lamb = data['lamb']
        self.n, self.dim = self.X.shape
        self.V = np.arange(self.n)
        self.training()

    def training(self):
        self.train_machine_error()
        # self.train_human_error()

    def get_subset(self, K):
        # print self.machine_err
        # err = np.sqrt(self.c) - np.absolute(self.machine_err)
        err = self.c - self.machine_err
        indices = np.argsort(err)
        # print indices
        return indices[:K]

    def train_machine_error(self):
        self.machine_err = self.fit_LR(self.X, self.Y)
        # self.w_machine_error, tmp = self.fit_LR(self.X, self.machine_err)

    # def train_human_error(self):
    #     self.w_human_error, tmp = self.fit_LR(self.X, np.sqrt(self.c))

    def fit_LR(self, X, Y):
        # w = LA.solve(X.T.dot(X) + self.lamb * np.eye(X.shape[1]), X.T.dot(Y))
        # err = np.absolute((X.dot(w) - Y))
        from sklearn import svm
        from sklearn.metrics import hinge_loss
        reg_par = float(1) / (2.0 * self.lamb * self.n)
        model = svm.LinearSVC(C=reg_par)
        # model = svm.SVC(kernel='linear', C=reg_par)
        model.fit(X, Y)
        y_pred = model.decision_function(X)
        w = model.coef_
        reg = self.lamb * (self.n) * np.dot(w, w.T)[0][0]
        loss = np.maximum(0,1-(y_pred*Y))
        # hinge_machine_loss = hinge_loss(Y, y_pred)
        # hinge_machine_loss *= y_pred.shape[0]
        # print np.sum(loss),hinge_machine_loss
        return reg + loss

class modular_distort_greedy:
    def __init__(self, data):
        self.X = data['X']
        self.Y = data['Y']
        self.c = data['c']
        self.lamb = data['lamb']
        self.svm_type = data['svm_type']
        self.g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb, 'svm_type' : self.svm_type})
        self.n, self.dim = self.X.shape
        self.V = np.arange(self.n)
        self.initialize()


    def initialize(self):
        # G_null = self.g.eval(np.array([]).astype(int))
        # self.null_val = max(0.0, - G_null)
        # prev = self.g.eval(np.array([0]))
        # # G_ascend = np.array([self.g.eval(np.arange(i + 1)) - self.g.eval(np.arange(i)) for i in self.V-10])
        # G_ascend = np.zeros(shape=self.V.shape[0])
        # for i in self.V-30:
        #     G_ascend[i] = self.g.eval(np.arange(i + 1)) - prev
        #     prev = G_ascend[i]
        # # G_ascend = np.array([self.g.eval(np.arange(i + 1)) - prev for i in self.V - 10])
        # self.w = np.array([np.max(np.array([0.0, G_ascend[i]])) for i in self.V]).flatten()
        self.w = np.array([self.c[i] for i in self.V])

    # def eval(self, subset):
    #     return self.null_val + self.w[subset].sum()  # ( )

    def get_c(self, subset):
        return np.array([int(i) for i in range(self.n) if i not in subset])

    def get_inc_arr(self, subset, rest_flag=False, subset_rest=None):
        if rest_flag:
            subset_c = subset_rest
        else:
            subset_c = self.get_c(subset)

        l = np.array([self.w[i] for i in subset_c]).flatten()
        return l


class G:
    def __init__(self, input):
        self.X = input['X']
        self.Y = input['Y']
        self.lamb = input['lamb']
        self.c = input['c']
        self.svm_type = input['svm_type']
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)
        self.init_data_str()

    def reset(self):
        self.init_data_str()

    def get_c(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def init_data_str(self):
        # self.yTy = self.Y.dot(self.Y)
        # self.xxT = self.X.T.dot(self.X)
        # self.xy = self.X.T.dot(self.Y.reshape(self.n, 1))
        self.c_S = 0
        self.curr_set_len = 0

    def get_hard_linear_svm_w_b(self, subset_c):
        from sklearn import svm
        x = self.X[subset_c]
        y = self.Y[subset_c]
        model = svm.LinearSVC(C=1000)
        model.fit(x, y)
        w = model.coef_
        b = model.intercept_
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]
        return reg

    def get_hard_linear_svm_w(self, subset_c):
        from sklearn import svm
        x = self.X[subset_c]
        y = self.Y[subset_c]
        model = svm.LinearSVC(fit_intercept=False, C=1000)
        model.fit(x, y)
        w = model.coef_
        b = model.intercept_
        assert (b == 0)
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]
        return reg

    def get_soft_linear_svm_w_b(self, subset_c):
        from sklearn import svm
        from sklearn.metrics import hinge_loss
        x = self.X[subset_c]
        y = self.Y[subset_c]
        reg_par = float(1) / (2.0 * self.lamb * subset_c.shape[0])
        model = svm.LinearSVC(C=reg_par)
        # model = svm.SVC(kernel='linear', C=reg_par)
        model.fit(x, y)
        y_pred = model.decision_function(x)
        w = model.coef_
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]
        # machine_loss = 0
        # for x_i, y_i in zip(x, y):
        #     pred = np.dot(y_i, np.dot(x_i, w.T) + b)
        #     pred = float(pred)
        #     machine_loss += max(0, 1 - pred)

        hinge_machine_loss = hinge_loss(y, y_pred)
        hinge_machine_loss *= y_pred.shape[0]
        return reg + hinge_machine_loss

    def get_soft_linear_svm_w(self, subset_c):
        from sklearn import svm
        from sklearn.metrics import hinge_loss
        x = self.X[subset_c]
        y = self.Y[subset_c]

        # model = svm.SVC(fit_intercept=False, kernel='poly', degree=2, C=0.01)
        reg_par = float(1)/(2.0*self.lamb*subset_c.shape[0])
        model = svm.LinearSVC(fit_intercept=False, C=reg_par)
        # print reg_par
        # model = SVC(C=reg_par, kernel='linear')
        # model.fit(x, y)
        model.fit(x,y)
        y_pred = model.decision_function(x)

        w = model.coef_
        b = model.intercept_
        assert (b == 0)
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w,w.T)[0][0]

        hinge_machine_loss = hinge_loss(y, y_pred)
        hinge_machine_loss *= y_pred.shape[0]
        return reg + hinge_machine_loss

    def get_soft_kernel_svm_w_b(self, subset_c):
        from sklearn import svm
        from sklearn.svm import NuSVC
        from sklearn.metrics import hinge_loss
        x = self.X[subset_c]
        y = self.Y[subset_c]

        hasplus=False
        hasminus=False
        for label in y:
            if label==1:
                hasplus=True
            if label==-1:
                hasminus=True

        # if((not hasplus) or (not hasminus)):
        #     return 1000000000


        # model = svm.SVC(kernel='poly', degree=2, C=0.01)
        reg_par = float(1) / (2.0 * self.lamb * subset_c.shape[0])
        # model = svm.LinearSVC(C=reg_par)
        # model = SVC(C=reg_par, kernel='polynomial', degree=2)
        # model = svm.SVC(kernel='poly', degree=2, C=reg_par)
        model = svm.SVC(C=30,gamma=0.0001)
        # model = svm.SVC(C=reg_par,kernel='poly',degree=2)
        model.fit(x, y)
        w = model.dual_coef_
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]
        y_pred = model.decision_function(x)
        hinge_machine_loss = hinge_loss(y, y_pred)
        hinge_machine_loss *= y_pred.shape[0]
        # print reg, hinge_machine_loss
        return reg + hinge_machine_loss

    def get_soft_kernel_svm_w(self, subset_c):
        from sklearn import svm
        from sklearn.svm import NuSVC
        from sklearn.metrics import hinge_loss
        generate_svmlight('data/data_dict_kernel', subset_c)

        # x = self.X[subset_c]
        # y = self.Y[subset_c]
        # model = svm.LinearSVC(C=0.0001)
        # model = svm.SVC(fit_intercept=False, kernel='poly', degree=2)
        reg_par =  float(1) / (2.0 * self.lamb)
        os.system('/home/nastaran/Downloads/bsvm-2.09/bsvm-train -s 1 -t 1 -d 2 -c ' + str(reg_par) + ' -g 0.5 data/data_dict_kernel_out.txt kernelmodel')

        # os.system(
        #     '/home/nastaran/Downloads/bsvm-2.09/bsvm-predict data/data_dict_kernel_out.txt kernelmodel kernel_result')

        # pred = get_f('kernel_result',x.shape[0])

        # model = SVC(C=0.01,kernel='polynomial',degree=2)
        # model = NuSVC(kernel='poly',degree=2)
        # model.fit(x, y)
        # y_pred = model.decision_function(x)

        # w = model.dual_coef_
        # w, sv = get_w('kernelmodel')
        # f = get_train_f('f',subset_c.shape[0])
        # y = y.astype('float')
        # myloss = 0
        # for data,label in zip(f,y):
        #     loss = label*data
        #     # loss = max(0,loss)
        #     myloss += loss

        # pred = y*f
        # margin = 1.0-pred
        # np.clip(margin, 0, None, out=margin)
        # myloss = np.sum(margin)
        # print w
        # print sv
        # b = model.intercept_
        # assert (b == 0)
        # reg = self.lamb * (subset_c.shape[0]) * LA.norm(w, ord=2)
        # first_reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]

        # def K(X1, X2):
        #     gamma = 0.5
        #     return (gamma * np.dot(X1, X2.T)) ** 2
        #
        # pred = y * (np.dot(K(x, sv), w.T).ravel())
        # # print pred
        # machine_loss = 1 - pred
        # np.clip(machine_loss, 0, None, out=machine_loss)
        # first_machine_loss = np.sum(machine_loss)
        machine_loss,reg_loss = get_hinge_loss('kernel_hinge_loss')

        # print first_machine_loss, machine_loss

        # print machine_loss
        reg_loss *= (self.lamb*subset_c.shape[0])
        # print first_reg, reg_loss
        # print reg_loss,reg
        # print machine_loss, reg_loss

        # hinge_machine_loss = hinge_loss(y, y_pred)
        # hinge_machine_loss *= y_pred.shape[0]
        # print machine_loss
        return reg_loss + machine_loss

    def update_data_str(self, elm):  # outsource elm to human
        # y = self.Y[elm]
        # x = self.X[elm].reshape(self.dim, 1)
        # self.yTy -= y * y
        # self.xxT -= x.dot(x.T)
        # self.xy -= y * x
        self.c_S += self.c[elm]
        self.curr_set_len += 1

    def give_inc(self, subset, elm):
        subset = np.append(subset, np.array([elm]).astype(int))
        subset_c = self.get_c(subset)
        if self.svm_type == 'hard_linear_with_offset':
            machine_error = self.get_hard_linear_svm_w_b(subset_c)

        if self.svm_type == 'hard_linear_without_offset':
            machine_error = self.get_hard_linear_svm_w(subset_c)

        if self.svm_type == 'soft_linear_with_offset':
            machine_error = self.get_soft_linear_svm_w_b(subset_c)

        if self.svm_type == 'soft_linear_without_offset':
            machine_error = self.get_soft_linear_svm_w(subset_c)

        if self.svm_type == 'soft_kernel_with_offset':
            machine_error = self.get_soft_kernel_svm_w_b(subset_c)

        if self.svm_type == 'soft_kernel_without_offset':
            machine_error = self.get_soft_kernel_svm_w(subset_c)

        # machine_error = self.my_soft_svm_w_b(subset_c)
        c_S = self.c_S + self.c[elm]

        # return -np.log(machine_error + c_S)
        return - machine_error

    def eval_curr(self, subset_c):
        if self.svm_type == 'hard_linear_with_offset':
            machine_error = self.get_hard_linear_svm_w_b(subset_c)

        if self.svm_type == 'hard_linear_without_offset':
            machine_error = self.get_hard_linear_svm_w(subset_c)

        if self.svm_type == 'soft_linear_with_offset':
            machine_error = self.get_soft_linear_svm_w_b(subset_c)

        if self.svm_type == 'soft_linear_without_offset':
            machine_error = self.get_soft_linear_svm_w(subset_c)

        if self.svm_type == 'soft_kernel_with_offset':
            machine_error = self.get_soft_kernel_svm_w_b(subset_c)

        if self.svm_type == 'soft_kernel_without_offset':
            machine_error = self.get_soft_kernel_svm_w(subset_c)
        # machine_error = self.my_soft_svm_w_b(subset_c)
        # return -np.log(machine_error + self.c_S)
        return - machine_error

    def get_inc_arr(self, subset, rest_flag=False, subset_rest=None):
        subset_c = self.get_c(subset)
        G_S = self.eval_curr(subset_c)

        if rest_flag:
            subset_c = subset_rest
        else:
            subset_c = self.get_c(subset)

        vec = []
        # G_S = self.eval_curr(subset_c)[0][0]
        # G_S = self.eval_curr(subset_c)


        for i in subset_c:
            vec.append(self.give_inc(subset, i) - G_S)

        return np.array(vec), subset_c

    def eval(self, subset=None):
        if subset.shape[0] == self.n:
            return -np.log(self.c.sum())

        subset_c = self.get_c(subset)

        if subset.size == 0:
            c_S = 0
        else:
            c_S = self.c[subset].sum()
        if self.svm_type == 'hard_linear_with_offset':
            machine_error = self.get_hard_linear_svm_w_b(subset_c)

        if self.svm_type == 'hard_linear_without_offset':
            machine_error = self.get_hard_linear_svm_w(subset_c)

        if self.svm_type == 'soft_linear_with_offset':
            machine_error = self.get_soft_linear_svm_w_b(subset_c)

        if self.svm_type == 'soft_linear_without_offset':
            machine_error = self.get_soft_linear_svm_w(subset_c)

        if self.svm_type == 'soft_kernel_with_offset':
            machine_error = self.get_soft_kernel_svm_w_b(subset_c)

        if self.svm_type == 'soft_kernel_without_offset':
            machine_error = self.get_soft_kernel_svm_w(subset_c)
        # machine_error = self.my_soft_svm_w_b(subset_c)
        # return -np.log(machine_error + c_S)
        return - machine_error
