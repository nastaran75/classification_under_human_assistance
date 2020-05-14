import numpy.random as rand
from process_image_data import *
import matplotlib.pyplot as plt
from random import sample



class Generate_human_error:

    def __init__(self, data_file):
        # print data_file
        self.data = load_data(data_file)
        if 'c' in self.data:
            del self.data['c']
            # del self.data['test']['c']
            del self.data['Pr_M']
            del self.data['Pr_H']
            del self.data['test']['Pr_M']
            del self.data['test']['Pr_H']
            del self.data['Pr_M_Alg']
            del self.data['test']['Pr_M_Alg']
            del self.data['Pr_M_gt']
            del self.data['Pr_H_gt']
            del self.data['y_h']
            del self.data['test']['y_h']
            del self.data['h']
            del self.data['test']['h']

        self.n, self.dim = self.data['X'].shape

    def normalize_features(self, delta=1):
        n, dim = self.data['X'].shape
        for feature in range(dim):
            self.data['X'][:, feature] = np.true_divide(self.data['X'][:, feature],
                                                        100 * LA.norm(self.data['X'][:, feature].flatten()))
            self.data['test']['X'][:, feature] = np.true_divide(self.data['test']['X'][:, feature], 100 * LA.norm(
                self.data['test']['X'][:, feature].flatten()))

        print np.max([LA.norm(x.flatten()) for x in self.data['X']])

    def white_Gauss(self, std=1, n=1, upper_bound=False, y_vec=None):
        init_noise = rand.normal(0, std, n)
        if upper_bound:
            return np.array([noise if noise / y < 0.3 else 0.1 * y for noise, y in zip(init_noise, y_vec)])
        else:
            return init_noise

    def data_independent_noise(self, list_of_std, upper_bound=False):
        self.data['c'] = {}
        self.data['test']['c'] = {}
        for std in list_of_std:
            self.data['c'][str(std)] = self.white_Gauss(std, self.data['Y'].shape[0], upper_bound, self.data['Y']) ** 2
            self.data['test']['c'][str(std)] = self.white_Gauss(std, self.data['test']['Y'].shape[0], upper_bound,
                                                                self.data['test']['Y']) ** 2

    def variable_std_Gauss(self, std_const, x):
        x_norm = LA.norm(x, axis=1).flatten()
        std_vector = std_const * np.reciprocal(x_norm)
        tmp = np.array([rand.normal(0, s, 1)[0] for s in std_vector])
        return tmp

    def data_dependent_noise(self, list_of_std):
        self.data['c'] = {}
        self.data['test']['c'] = {}
        for std in list_of_std:
            self.data['c'][str(std)] = self.variable_std_Gauss(std, self.data['X']) ** 2
            self.data['test']['c'][str(std)] = self.variable_std_Gauss(std, self.data['test']['X']) ** 2

    def modify_y_values(self):
        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y).shape[0]

        def map_range(v, l, h, l_new, h_new):
            return float(v - l) * ((h_new - l_new) / float(h - l)) + l_new

        num_cat = get_num_category(self.data['Y'], self.data['test']['Y'])
        print num_cat
        self.data['Y'] = np.array([map_range(i, 0, 1, float(1) / num_cat, 1) for i in self.data['Y']]).flatten()
        self.data['test']['Y'] = np.array(
            [map_range(i, 0, 1, float(1) / num_cat, 1) for i in self.data['test']['Y']]).flatten()

    def get_discrete_noise(self, p, num_cat):
        m = 10
        c = []
        for sample in range(self.n):
            if False:
                sum = 0
                for i in range(m):
                    x = np.random.uniform(0, 1)
                    if x < p:
                        sum += (float(1) / num_cat) ** 2
                c.append(float(sum) / m)
            else:
                c.append(((float(1) / num_cat) ** 2) * p)
        return np.array(c)

    def discrete_noise(self, list_of_p):
        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y).shape[0]

        num_cat = get_num_category(self.data['Y'], self.data['test']['Y'])
        self.data['c'] = {}
        self.data['test']['c'] = {}

        for p in list_of_p:
            self.data['c'][str(p)] = self.get_discrete_noise(p, num_cat)
            self.data['test']['c'][str(p)] = self.get_discrete_noise(p, num_cat)

    def vary_discrete_old_data_format(self, noise_ratio, list_of_high_err_const):
        def nearest(i):
            return np.argmin(self.data['dist_mat'][i])

        n = self.data['X'].shape[0]
        indices = np.arange(n)
        random.shuffle(indices)
        self.data['low'] = {}
        self.data['c'] = {}
        self.data['test']['c'] = {}

        for high_err_const in list_of_high_err_const:
            num_low = int(high_err_const * n)
            self.data['low'][str(high_err_const)] = indices[:num_low]
            self.data['c'][str(high_err_const)] = np.array(
                [0.0001 if i in self.data['low'][str(high_err_const)] else 0.25 for i in range(n)])
            self.data['test']['c'][str(high_err_const)] = np.array(
                [0.0001 if nearest(i) in self.data['low'][str(high_err_const)] else 0.5 for i in
                 range(self.data['test']['X'].shape[0])])

    def vary_discrete_3(self, list_of_noise_ratio):
        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y).shape[0]

        def nearest(i):
            return np.argmin(self.data['dist_mat'][i])

        self.normalize_features()
        num_cat = get_num_category(self.data['Y'], self.data['test']['Y'])
        high_err_const = 45
        n = self.data['X'].shape[0]
        indices = np.arange(n)
        random.shuffle(indices)
        err = ((float(1) / num_cat) ** 2) / 50
        self.data['low'] = {}
        self.data['c'] = {}
        self.data['test']['c'] = {}

        # list_of_low_indices = []
        for noise_ratio in list_of_noise_ratio:
            num_low = int(noise_ratio * n)
            # print num_low
            self.data['low'][str(noise_ratio)] = indices[:num_low]
            self.data['c'][str(noise_ratio)] = np.array(
                [err if i in self.data['low'][str(noise_ratio)] else high_err_const * err for i in range(n)])
            print 'c min', np.min(self.data['c'][str(noise_ratio)])
            print 'c max', np.max(self.data['c'][str(noise_ratio)])
            self.data['test']['c'][str(noise_ratio)] = np.array(
                [err if nearest(i) in self.data['low'][str(noise_ratio)] else high_err_const * err for i in
                 range(self.data['test']['X'].shape[0])])

    def vary_discrete_3_v2(self, list_of_class_indices):
        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y).shape[0]

        def nearest(i):
            return np.argmin(self.data['dist_mat'][i])

        self.normalize_features()
        num_cat = get_num_category(self.data['Y'], self.data['test']['Y'])
        err = ((float(1) / num_cat) ** 2) / 50
        self.data['c'] = {}
        self.data['test']['c'] = {}
        list_of_high_indices = []
        c = np.ones(self.data['X'].shape[0]) * err

        for class_index in list_of_class_indices:
            class_label = float(class_index + 1) / num_cat
            new_indices = np.where(self.data['Y'] == class_label)[0]
            list_of_high_indices.extend(list(new_indices))
            err_upper = class_label ** 2
            c[new_indices] = err_upper
            self.data['c'][str(class_index)] = np.copy(c)
            print 'c min', np.min(self.data['c'][str(class_index)])
            print 'c max', np.max(self.data['c'][str(class_index)])
            plt.plot(self.data['c'][str(class_index)])
            plt.show()
            plt.close()
            self.data['test']['c'][str(class_index)] = np.ones(self.data['test']['X'].shape[0]) * err

    def vary_discrete(self, list_of_frac, file_name):

        def nearest(i):
            return np.argmin(self.data['dist_mat'][i])

        self.normalize_features()
        n = self.data['X'].shape[0]
        indices = np.arange(n)
        random.shuffle(indices)

        self.data['low'] = {}
        self.data['c'] = {}
        self.data['test']['c'] = {}

        for frac in list_of_frac:
            num_low = int(frac * n)
            self.data['low'][str(frac)] = indices[:num_low]

            if file_name == 'Umessidor':
                self.data['c'][str(frac)] = np.array(
                    [0.0001 if i in self.data['low'][str(frac)] else 0.4 for i in range(n)])
                self.data['test']['c'][str(frac)] = np.array(
                    [0.0001 if nearest(i) in self.data['low'][str(frac)] else 0.4 for i in
                     range(self.data['test']['X'].shape[0])])

            if file_name == 'Ustare11':
                self.data['c'][str(frac)] = np.array(
                    [0.0001 if i in self.data['low'][str(frac)] else 0.1 for i in range(n)])
                self.data['test']['c'][str(frac)] = np.array(
                    [0.0001 if nearest(i) in self.data['low'][str(frac)] else 0.25 for i in
                     range(self.data['test']['X'].shape[0])])

            if file_name == 'Ustare5':
                self.data['c'][str(frac)] = np.array(
                    [0.0001 if i in self.data['low'][str(frac)] else 0.1 for i in range(n)])
                self.data['test']['c'][str(frac)] = np.array(
                    [0.0001 if nearest(i) in self.data['low'][str(frac)] else 0.1 for i in
                     range(self.data['test']['X'].shape[0])])

    def variable_std_dirichlet(self, x_tr, y_tr, std, file_name):

        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y)

        cats = get_num_category(self.data['Y'], self.data['test']['Y'])

        if file_name == 'messidor':
            prob = [[6, 3, 1], [2, 6, 2], [1, 3, 6]]

        if file_name == 'stare5':
            prob = [[6, 3, 2, 2, 1, 1], [3, 6, 2, 2, 1, 1], [2, 3, 6, 2, 1, 1], [1, 2, 3, 6, 2, 1], [1, 1, 2, 3, 6, 2],
                    [1, 1, 2, 2, 3, 6]]

        if file_name == 'stare11':
            prob = [[6, 3, 2, 2, 1], [3, 6, 2, 2, 1], [1, 3, 6, 2, 2], [1, 2, 3, 6, 2],
                    [1, 2, 2, 3, 6]]
        tmp = []
        for idx, data in enumerate(x_tr):
            label = y_tr[idx]
            trueindex = np.argwhere(label == cats)
            assert trueindex.size == 1
            trueindex = trueindex.flatten()[0]
            error = 0
            std_vector = np.random.dirichlet(prob[trueindex])
            for idx, probability in enumerate(std_vector):
                error += (probability * ((cats[idx] - cats[trueindex]) ** 2))
            tmp.append(error)

        tmp = np.array(tmp)
        tmp = tmp.reshape(tmp.shape[0])

        return tmp

    def dirichlet(self, list_of_p, file_name):

        self.data['c'] = {}
        self.data['test']['c'] = {}
        for std in list_of_p:
            self.data['c'][str(std)] = self.variable_std_dirichlet(self.data['X'], self.data['Y'], std, file_name)
            self.data['test']['c'][str(std)] = self.variable_std_dirichlet(self.data['test']['X'],
                                                                           self.data['test']['Y'], std, file_name)

    def generate_variable_human_prediction(self, Y, list_of_std, threshold):
        c = {}
        y_h = {}
        h = {}
        Pr_H = {}
        Pr_H_gt = {}
        doctors_label = {}
        doctor_label_thresholded = {}
        num_doctors = 10
        scale = 4  # 3
        def descrete_label(cont):
            if cont > threshold:
                return 1
            else:
                return -1

        # print h_threshold,h_test_threshold

        for std in list_of_std:
            h_threshold = np.random.normal(threshold, std, (Y.shape[0],num_doctors))
            doctors_h = np.zeros((Y.shape[0], num_doctors))
            y_h[str(std)] = np.ones(shape=Y.shape[0])
            c[str(std)] = np.ones(shape=Y.shape[0])
            h[str(std)] = np.ones(shape=Y.shape[0])
            doctor_label_thresholded = np.zeros((Y.shape[0], num_doctors))
            Pr_H[str(std)] = np.zeros(shape=Y.shape[0],dtype='int')
            Pr_H_gt[str(std)] = np.zeros(shape=Y.shape[0])
            # h = (self.data['Y']*scale - sample(list,1))
            for idx, label in enumerate(Y):
                if label > threshold:
                    label = 1
                else:
                    label = -1
                doctors_h[idx] = (Y[idx] - h_threshold[idx]) * scale


                for doc_idx, doc_h in enumerate(doctors_h[idx]):
                    if doc_h>0:
                        doctor_label_thresholded[idx][doc_idx] = 1
                    else:
                        doctor_label_thresholded[idx][doc_idx] = -1

                Pr_H_gt[str(std)][idx] = np.mean(np.absolute(doctor_label_thresholded - label))/2.0
                A_idx = []
                while len(A_idx) < num_doctors/2:
                    num = np.random.randint(0, num_doctors)
                    if num not in A_idx:
                        A_idx.append(num)
                # print A_idx
                A = np.array([doc_label for doc_idx, doc_label in enumerate(doctor_label_thresholded[idx]) if doc_idx in A_idx])
                B = np.array(
                    [doc_label for doc_idx, doc_label in enumerate(doctor_label_thresholded[idx]) if doc_idx not in A_idx])
                # print A
                # print B
                A_mean = np.sum(A)
                B_mean = np.sum(B)

                if A_mean *B_mean <0 : #doctors disagree
                    Pr_H[str(std)][idx] = 1  # disagreement
                else:
                    Pr_H[str(std)][idx] = 0  # agreement

                final_label = np.sum(doctor_label_thresholded[idx])
                if final_label>0:
                    final_label = 1
                else:
                    final_label = -1

                # if human_label > threshold:
                #     human_label_thresholded = 1
                # else:
                #     human_label_thresholded = -1

                y_h[str(std)][idx] = final_label

                #
                # if h_human[idx] * label < 0:
                #     y_h[str(std)][idx] = -label
                # else:
                #     y_h[str(std)][idx] = label
                final_h = np.mean(doctors_h[idx])
                c[str(std)][idx] = np.maximum(0, 1 - (label * final_h))
                h[str(std)][idx] = final_h

        print c, h, y_h, Pr_H, Pr_H_gt
        return c, h, y_h, Pr_H, Pr_H_gt


    def estimated_uncertainty_generate_variable_human_prediction(self, Y, list_of_std, threshold):
        c = {}
        y_h = {}
        h = {}
        Pr_H = {}
        Pr_H_gt = {}
        doctors_label = {}
        doctor_label_thresholded = {}
        num_doctors = 10

        def descrete_label(cont):
            if cont > threshold:
                return 1
            else:
                return -1

        def compute_h(cont_label):
            return (2.0 * (cont_label - cats[0]) / (cats[cats.size-1] - cats[0])) - 0.7

        cats = np.unique(Y)
        # prob = [[0.2, 0.1, 0.1], [0.1, 0.15, 0.1], [0.1, 0.1, 0.15]] #lamb 1 human error less than machine
        prob = [[0.2, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.2]] #lamb 1 human error more than machine

        for std in list_of_std:
            y_h[str(std)] = np.zeros(shape=Y.shape, dtype='int')
            c[str(std)] = np.zeros(shape=Y.shape)
            h[str(std)] = np.zeros(shape=Y.shape)
            Pr_H[str(std)] = np.zeros(shape=Y.shape)
            Pr_H_gt[str(std)] = np.zeros(shape=Y.shape)
            doctors_label[str(std)] = np.zeros((Y.shape[0], num_doctors))
            doctor_label_thresholded[str(std)] = np.zeros((Y.shape[0],num_doctors))

            for idx, label in enumerate(Y):

                thresholded_label = descrete_label(label)
                trueindex = np.argwhere(label == cats)
                assert trueindex.size == 1
                trueindex = trueindex.flatten()[0]
                std_vector = np.random.dirichlet(prob[trueindex], size=num_doctors)
                hmap = {cats[0]: -2.0, cats[1]: 1.5, cats[2]: 2.0}

                for std_idx, doctor_prob in enumerate(std_vector):
                    doctors_label[str(std)][idx][std_idx] = (np.random.choice(cats, 1, p=doctor_prob)[0])
                    for prob_idx,probability in enumerate(doctor_prob):
                        c[str(std)][idx] += probability * np.maximum(0, 1 - (hmap[cats[prob_idx]] * thresholded_label)) * 0.01

                c[str(std)][idx] /= num_doctors

                doctor_label_thresholded[str(std)][idx] = [descrete_label(doc_label) for doc_label in doctors_label[str(std)][idx]]

                Pr_H_gt[str(std)][idx] = np.mean(np.absolute(doctor_label_thresholded[str(std)][idx] - Y[idx]))/2.0

                A_idx = []
                while len(A_idx) < num_doctors / 2:
                    num = np.random.randint(0, num_doctors)
                    if num not in A_idx:
                        A_idx.append(num)

                A = np.array([A_label for doc_idx, A_label in enumerate(doctors_label[str(std)][idx]) if doc_idx in A_idx])
                B = np.array([B_label for doc_idx, B_label in enumerate(doctors_label[str(std)][idx]) if doc_idx not in A_idx])

                A_mean = descrete_label(np.mean(A))
                B_mean = descrete_label(np.mean(B))

                if A_mean != B_mean:
                    Pr_H[str(std)][idx] = 1  # disagreement
                else:
                    Pr_H[str(std)][idx] = 0  # agreement

                final_label = np.mean(doctors_label[str(std)][idx])
                final_label_thresholded = descrete_label(final_label)
                h[str(std)][idx] = compute_h(final_label)
                y_h[str(std)][idx] = final_label_thresholded


        # print c, h, y_h, Pr_H, Pr_H_gt
        print c

        return c, h, y_h, Pr_H, Pr_H_gt

    def estimated_uncertainty(self, list_of_std, threshold):
        self.data['c'], self.data['h'], self.data['y_h'], self.data['Pr_H'], self.data['Pr_H_gt']= \
            self.estimated_uncertainty_generate_variable_human_prediction(self.data['Y'], list_of_std, threshold)

        self.data['test']['c'], self.data['test']['h'], self.data['test']['y_h'], self.data['test']['Pr_H'], self.data['test']['Pr_H_gt'] = \
            self.estimated_uncertainty_generate_variable_human_prediction(self.data['test']['Y'], list_of_std,
                                                                          threshold)

        # self.data['c'], self.data['h'], self.data['y_h'], self.data['Pr_H'], self.data['Pr_H_gt'] = \
        #     self.generate_variable_human_prediction(self.data['Y'],list_of_std,threshold)
        #
        # self.data['test']['c'], self.data['test']['h'], self.data['test']['y_h'], self.data['test']['Pr_H'], self.data['test']['Pr_H_gt'] = \
        #     self.generate_variable_human_prediction(self.data['test']['Y'],list_of_std,threshold)

        for std in list_of_std:
            from sklearn.neural_network import MLPClassifier
            lamb = 1
            X = self.data['X']
            model = MLPClassifier(max_iter=300)
            model.fit(X, self.data['Pr_H'][str(std)])
            self.data['test']['Pr_H'][str(std)] = model.predict(self.data['test']['X'])

        self.data['Pr_M'] = {}
        self.data['test']['Pr_M'] = {}
        self.data['Pr_M_Alg'] = {}
        self.data['test']['Pr_M_Alg'] = {}
        self.data['Pr_M_gt'] = {}

        for std in list_of_std:
            self.data['Pr_M_gt'][str(std)] = self.compute_Pr_M_gt(threshold,lamb)
            self.data['Pr_M_Alg'][str(std)], self.data['test']['Pr_M_Alg'][str(std)] = self.compute_triage_alg(threshold,std,lamb)
            self.data['Pr_M'][str(std)] ,self.data['test']['Pr_M'][str(std)] = self.compute_triage_estimated(threshold,std,lamb)

        # print self.data['Pr_M']
        # print self.data['test']['Pr_M']
        # print self.data['Pr_M_gt']
        # # print self.data['test']['Pr_M_gt']
        print self.data['Pr_M_Alg']
        print self.data['test']['Pr_M_Alg']
        # self.data['c'] = self.data['Pr_M_Alg']

        # # print self.data['c']

    def compute_Pr_M_gt(self,threshold,lamb):

        def descrete_label(cont):
            if cont > threshold:
                return 1
            else:
                return -1

        from sklearn.svm import SVC
        reg_par = float(1) / (2.0 * lamb * self.data['X'].shape[0])
        # model = SVC(gamma=0.001, C=reg_par)
        model = SVC(kernel='linear', C=reg_par)
        model.fit(self.data['X'], self.get_labels(self.data['Y'],threshold))
        h = model.decision_function(self.data['X'])
        rank = np.argsort(-np.absolute(h))
        # loss = np.maximum(0, 1 - (h * self.data['Y']))
        error = np.zeros(self.data['X'].shape[0])
        for i in range(2000):
            R = 0
            for idx in range(self.data['X'].shape[0]):
                doctor_grade = np.random.choice([-1,1],1, [0.7,0.3])
                if doctor_grade == 1:
                    R += 1
            # print R
            ranking = rank[:R]
            m = np.zeros(self.data['X'].shape[0])
            for idx, m_val in enumerate(m):
                if idx in ranking:
                    m[idx] = 1
                else:
                    m[idx] = -1
                error[idx] += np.absolute(descrete_label(self.data['Y'][idx]) - m[idx])
        # print error
        return error/4000

    def compute_triage_alg(self, threshold,std,lamb):
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        X = self.data['X']
        reg_par = float(1) / (2.0 * lamb * X.shape[0])
        # machine_model = SVC(C=reg_par, gamma=0.001)
        machine_model = SVC(C=reg_par,kernel='linear')
        machine_model.fit(X, self.get_labels(self.data['Y'], threshold))
        train_pred = machine_model.predict(X)
        print np.mean(train_pred == self.get_labels(self.data['Y'], threshold))
        print np.mean(self.data['y_h'][str(std)] == self.get_labels(self.data['Y'], threshold))

        h = np.absolute(machine_model.decision_function(X))
        h_test = np.absolute(machine_model.decision_function(self.data['test']['X']))


        return 1.0 / (1.0 + np.exp(h)), 1.0 / (1.0 + np.exp(h_test))
        # return 1.0 / (1.0 + np.absolute(h)), 1.0 / (1.0 + np.absolute(h_test))

    def compute_triage_estimated(self,threshold,std,lamb):
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        X = self.data['X']
        reg_par = float(1) / (2.0 * lamb * X.shape[0])
        # machine_model = SVC(C=reg_par, gamma=0.001)
        machine_model = SVC(C=reg_par, kernel='linear')
        machine_model.fit(X, self.get_labels(self.data['Y'], threshold))
        train_pred = machine_model.predict(X)
        Y = np.zeros(X.shape[0], dtype='int')
        for idx, item in enumerate(self.data['y_h'][str(std)]):
            if item != train_pred[idx]:  # disagreement
                Y[idx] = 1  # disagreement

        model = MLPClassifier(max_iter=300)
        model.fit(X, Y)

        return Y, model.predict(self.data['test']['X'])

    def get_labels(self, cont_y,threshold):
        y = np.zeros(cont_y.shape)
        cntplus = 0
        cntminus = 0
        for idx, label in enumerate(cont_y):
            if label > threshold:
                y[idx] = 1
                cntplus += 1
            else:
                y[idx] = -1
                cntminus += 1
        print 'positive, negative'
        print cntplus, cntminus
        # print y
        return y

    def save_data(self, data_file):
        save(self.data, data_file)


def generate_human_error(file_name_list):
    list_of_option = ['random_noise', 'vary_std_noise', 'norm_rand_noise', \
                      'discrete', 'vary_discrete', 'vary_discrete_old', 'vary_discrete_3', 'dirichlet',
                      'variable_human']

    datasets = ['stare5', 'stare11', 'messidor', 'Ustare5', 'Ustare11', 'Umessidor']

    for file_name in file_name_list:
        assert file_name in datasets

        if file_name.startswith('U'):
            option = 'vary_discrete'
        else:
            # option = 'dirichlet'
            # option = 'variable_human'
            option = 'estimated_uncertainty'

        if option == 'discrete' or option == 'dirichlet':
            list_of_std = [0.05, 0.08, 0.1, 0.2]
        else:
            # if 'vary_discrete' in option:
            #     list_of_std = [0.2, 0.4, 0.6, 0.8]
            # else:
            #     list_of_std = [(i + 1) * 0.01 for i in range(9)]
            # list_of_std = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1]#[0.2,0.3,0.4,0.5]#[0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5]
            list_of_std =[1]# [0.1,0.2,0.3,0.4]

        data_file = 'data/data_dict_' + file_name

        obj = Generate_human_error(data_file)

        if option == 'variable_human':
            # obj.generate_variable_human_prediction(list_of_std,0.5)
            obj.dirichlet_generate_variable_human_prediction(list_of_std, 0.5)

        if option == 'estimated_uncertainty':
            obj.estimated_uncertainty(list_of_std, 0.65)

        if option == 'dirichlet':
            obj.dirichlet(list_of_std, file_name)

        if option == 'random_noise':
            obj.data_independent_noise(list_of_std)

        if option == 'vary_std_noise':
            obj.data_dependent_noise(list_of_std)

        if option == 'norm_rand_noise':
            obj.normalize_features(delta=0.001)
            obj.data_independent_noise(list_of_std, upper_bound=True)

        if option == 'discrete':
            obj.discrete_noise(list_of_std)

        if option == 'vary_discrete':
            obj.vary_discrete(list_of_std, file_name)

        if option == 'vary_discrete_old':
            obj.vary_discrete_old_data_format(0.5, list_of_std)

        if option == 'vary_discrete_3':
            obj.vary_discrete_3_v2(list_of_std)

        obj.save_data(data_file)


def main():
    # generating human error for image datasets.
    file_name_list = sys.argv[1:]
    generate_human_error(file_name_list)


if __name__ == "__main__":
    main()
