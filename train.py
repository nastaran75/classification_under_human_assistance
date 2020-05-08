import sys
import time

from myutil import *
from algorithms import triage_human_machine
import parser


def parse_command_line_input(dataset):
    list_of_option = ['greedy', 'RLSR_Reg', 'distort_greedy', 'kl_triage', 'diff_submod']

    list_of_real = ['messidor', 'stare5', 'stare11', 'hatespeech',
                    'Ustare5', 'Ustare11', 'Umessidor']
    list_of_synthetic = ['sigmoid', 'gauss', 'Ugauss', 'Usigmoid', 'Wgauss', 'Wsigmoid','linear',
                         'kernel','hard_linear','linear_with_offset','hard_linear_with_offset','kernel_with_offset','quan_linear','quan_kernel']

    assert (dataset in list_of_real or dataset in list_of_synthetic)

    if dataset in ['sigmoid', 'gauss']:
        list_of_std = [0.001]
        list_of_lamb = [0.005]

    if dataset == 'messidor':
        list_of_std = [0.1]#[0.1,0.2,0.3,0.4,0.5]#[0.5]#[0.2,0.3,0.4,0.5]#[0.15]#[0.05,0.08,0.1,0.15]#0.2,0.3]#,0.4,0.5]#[0.1]
        list_of_lamb = [1.0,10,100]
        list_of_option = ['kl_triage', 'stochastic_distort_greedy']#,'distort_greedy']

    if dataset == 'stare5':
        list_of_std = [0.1]
        list_of_lamb = [1.0]
        list_of_option = ['kl_triage', 'distort_greedy']

    if dataset == 'stare11':
        list_of_std = [0.1]
        # list_of_lamb = [1.0]
        list_of_lamb = [1.0]
        list_of_option = ['kl_triage', 'distort_greedy']

    if dataset == 'hatespeech':
        list_of_std = [0.0]
        list_of_lamb = [0.01]

    if dataset in ['Usigmoid', 'Ugauss']:
        list_of_std = [0.01, 0.02, 0.03, 0.04, 0.05]
        list_of_lamb = [0.005]
        list_of_option = ['greedy']

    if dataset in ['Ustare5', 'Ustare11', 'Umessidor']:
        list_of_std = [.2, .4, .6, .8]
        list_of_lamb = [1]
        list_of_option = ['greedy']

    if dataset in ['Wgauss', 'Wsigmoid']:
        list_of_std = [0.001]
        list_of_lamb = [0.001]
        list_of_option = ['greedy']
    if dataset in ['linear','kernel','hard_linear','linear_with_offset','hard_linear_with_offset','kernel_with_offset','quan_linear']:
        list_of_std = [0.001]
        # list_of_lamb = [0.001]
        list_of_lamb = [1.0]
        list_of_option = ['kl_triage','distort_greedy']

    if dataset in ['quan_linear','quan_kernel']:
        list_of_std = [1.0]
        # list_of_lamb = [0.001]
        list_of_lamb = [1.0]
        list_of_option = ['kl_triage','distort_greedy']

    return list_of_option, list_of_std, list_of_lamb




class eval_triage:
    def __init__(self, data_file, list_of_K, list_of_std, list_of_lamb, list_of_option,real_flag=None, real_wt_std=None):
        self.data = load_data(data_file)
        self.real = real_flag
        self.real_wt_std = real_wt_std
        self.list_of_K = list_of_K
        self.list_of_std = list_of_std
        self.list_of_lamb = list_of_lamb
        self.list_of_option = list_of_option
        self.threshold = 0.5

    def get_labels(self,cont_y):
        y = np.zeros(cont_y.shape)
        cntplus = 0
        cntminus = 0
        for idx, label in enumerate(cont_y):
            if label > self.threshold:
                y[idx] = 1
                cntplus += 1
            else:
                y[idx] = -1
                cntminus += 1
        print 'positive, negative'
        print cntplus, cntminus
        # print y
        return y

    def eval_loop(self, param, res_file, option,svm_type,is_stochastic):
        print option
        res = load_data(res_file, 'ifexists')
        # print res
        for std in param['std']:
            if self.real:
                data_dict = self.data
                triage_obj = triage_human_machine(data_dict, self.real)

            else:
                if self.real_wt_std:
                    print self.data.keys()
                    # print self.data['c'].keys()

                    # y_tr = self.get_labels(self.data['Y'],0.4)
                    # data_dict = self.data[str(std)]
                    data_dict = {'X': self.data['X'], 'Y': self.data['Y'], 'c': self.data['c'][str(std)]}
                    triage_obj = triage_human_machine(data_dict, self.real_wt_std)

                else:
                    test = {'X': self.data.Xtest, 'Y': self.data.Ytest,
                            'human_pred': self.data.human_pred_test[str(std)]}
                    data_dict = {'test': test, 'dist_mat': self.data.dist_mat, 'X': self.data.Xtrain,
                                 'Y': self.data.Ytrain, 'human_pred': self.data.human_pred_train[str(std)]}
                    triage_obj = triage_human_machine(data_dict, False)

            if str(std) not in res:
                res[str(std)] = {}

            for K in param['K']:
                if str(K) not in res[str(std)]:
                    res[str(std)][str(K)] = {}

                for lamb in param['lamb']:
                    if str(lamb) not in res[str(std)][str(K)]:
                        res[str(std)][str(K)][str(lamb)] = {}

                    print 'std-->', std, 'K--> ', K, ' Lamb--> ', lamb
                    res_dict = triage_obj.algorithmic_triage({'K': K, 'lamb': lamb, 'svm_type': svm_type,'is_stochastic' : is_stochastic},
                                                             optim=option)
                    res[str(std)][str(K)][str(lamb)][option] = res_dict
                    save(res, res_file)

    def check_w(self, res_file, svm_type):
        res = load_data(res_file, 'ifexists')
        # list_of_K = [0.2, 0.4, 0.6, 0.8]
        list_of_K =[0.4]#[0.1, 0.2, 0.3, 0.4,0.5]#, 0.5, 0.6, 0.7, 0.8, 0.9]
        # list_of_K = [0.1]

        for option in self.list_of_option:
            print option


            for std in self.list_of_std:
                print std
                for K in list_of_K:
                    print K
                    for lamb, ind in zip(self.list_of_lamb, range(len(self.list_of_lamb))):
                        if str(std) in res and str(K) in res[str(std)] and str(lamb) in res[str(std)][
                            str(K)] and option in res[str(std)][str(K)][str(lamb)]:
                            res[str(std)][str(K)][str(lamb)][option] = {}

                        if str(std) not in res:
                            res[str(std)] = {}
                        if str(K) not in res[str(std)]:
                            res[str(std)][str(K)] = {}
                        if str(lamb) not in res[str(std)][str(K)]:
                            res[str(std)][str(K)][str(lamb)] = {}

                        print self.data.keys()
                        # local_data = self.data[str(std)]
                        # local_data = self.data
                        Y = self.get_labels(self.data['Y'])
                        local_data = {'X': self.data['X'], 'Y': Y, 'c': self.data['c'][str(std)]}

                        triage_obj = triage_human_machine(local_data, True)
                        res_dict = triage_obj.algorithmic_triage({'K': K, 'lamb': lamb, 'svm_type': svm_type}, optim=option)
                        res[str(std)][str(K)][str(lamb)][option] = res_dict
                        save(res, res_file)

    def split_res_over_K(self, data_file, res_file, unified_K, option):
        res = load_data(res_file)
        for std in self.list_of_std:
            if str(std) not in res:
                res[str(std)] = {}

            for K in self.list_of_K:
                if str(K) not in res[str(std)]:
                    res[str(std)][str(K)] = {}

                for lamb in self.list_of_lamb:
                    if str(lamb) not in res[str(std)][str(K)]:
                        res[str(std)][str(K)][str(lamb)] = {}

                    if option not in res[str(std)][str(K)][str(lamb)]:
                        res[str(std)][str(K)][str(lamb)][option] = {}

                        if K != unified_K:
                            print res[str(std)].keys()
                            res_dict = res[str(std)][str(unified_K)][str(lamb)][option]
                            if res_dict:
                                res[str(std)][str(K)][str(lamb)][option] = self.get_res_for_subset(data_file, res_dict,
                                                                                                   lamb, K)
                                print res[str(std)][str(K)][str(lamb)][option]['subset'].shape
        save(res, res_file)


    def get_res_for_subset(self, data_file, res_dict, lamb, K):
        data = load_data(data_file)
        curr_n = int(data['X'].shape[0] * K)
        print 'curr_n'
        print curr_n, int(data['X'].shape[0]),  K
        subset_tr = res_dict['subset'][:curr_n]
        # w = self.get_optimal_pred(data, subset_tr, lamb)
        return {'subset': subset_tr}



def main():
    my_parser = parser.opts_parser()
    args = my_parser.parse_args()
    args = vars(args)
    list_of_file_names = [args['dataset']]
    svm_type = args['svm_type']
    # is_stochastic = args['stochastic']
    # if(is_stochastic):
    #     distort_greedy_type = 'sdg'
    # else:
    #     distort_greedy_type = 'dg'


    start = time.time()

    for file_name in list_of_file_names:
        print 'training ' + file_name
        list_of_option, list_of_std, list_of_lamb = parse_command_line_input(file_name)
        data_file = 'data/data_dict_' + file_name
        print data_file

        if not os.path.exists('Results'):
            os.mkdir('Results')
        res_file = 'Results/' + file_name + '_' + svm_type + '_res'
        # if os.path.exists(res_file+'.pkl'):
        #     os.remove(res_file+'.pkl')
        list_of_K = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]#, 0.45,0.5]#,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]

        if file_name in ['messidor','kernel','quan_kernel','Wgauss', 'Wsigmoid','linear','hard_linear','linear_with_offset','hard_linear_with_offset']:
            obj = eval_triage(data_file,list_of_K, list_of_std, list_of_lamb, list_of_option)
            obj.check_w(res_file=res_file,svm_type=svm_type)
            for option in list_of_option:
                if option not in ['diff_submod', 'RLSR', 'RLSR_Reg']:
                    unified_K = 0.4
                    obj.split_res_over_K(data_file, res_file, unified_K, option)
        # else:
        #     DG_T = 5
        #     if file_name == 'gauss':
        #         DG_T = 10
        #     if file_name == 'sigmoid':
        #         DG_T = 20
        #
        #     for option in list_of_option:
        #         if option in ['diff_submod', 'RLSR', 'RLSR_Reg']:
        #             list_of_K = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #
        #         else:
        #             list_of_K = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]
        #             # list_of_K = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #
        #         param = {'std': list_of_std, 'K': list_of_K, 'lamb': list_of_lamb, 'DG_T': DG_T}
        #         obj = eval_triage(data_file, real_wt_std=True)
        #         obj.check_w(res_file,list_of_std=param['std'],list_of_lamb=param['lamb'],option=option,svm_type=svm_type)


    end = time.time()
    print end - start


if __name__ == "__main__":

    main()
