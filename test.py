import sys
import matplotlib
# from brewer2mpl import brewer2mpl
from myutil import *
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from plot_util import latexify
import parser


def parse_command_line_input(dataset):
    list_of_option = ['greedy', 'RLSR_Reg', 'distort_greedy', 'kl_triage', 'diff_submod']
    list_of_real = ['messidor', 'stare5', 'stare11', 'hatespeech',
                    'Ustare5', 'Ustare11', 'Umessidor']

    list_of_synthetic = ['sigmoid', 'gauss', 'Ugauss', 'Usigmoid', 'Wgauss', 'Wsigmoid',
                         'linear', 'kernel', 'hard_linear', 'linear_with_offset', 'hard_linear_with_offset',
                         'kernel_with_offset', 'quan_linear', 'quan_kernel']

    assert (dataset in list_of_real or dataset in list_of_synthetic)

    if dataset in ['sigmoid', 'gauss']:
        list_of_std = [0.001]
        list_of_lamb = [0.005]

    if dataset == 'messidor':
        # list_of_std = [0.1]
        list_of_std = [0.1]#[0.1,0.2,0.3,0.4,0.5]#[0.15,0.2,0.25,0.3]#[0.6,0.7,0.8,0.9,1]#[0.5]#[0.2,0.3,0.4,0.5]#[0.15]#[0.05, 0.08, 0.1, 0.15]  # 0.2, 0.3]#, 0.4, 0.5]
        list_of_lamb = [1.0,10,100]#[1.0,0.1]#[10,1.0,0.1,0.01,0.001]
        list_of_option = ['kl_triage','stochastic_distort_greedy']#, 'distort_greedy']#,'stochastic_distort_greedy']

    if dataset == 'stare5':
        list_of_std = [0.1]
        list_of_lamb = [0.5]

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

    if dataset in ['linear', 'kernel', 'hard_linear', 'linear_with_offset', 'hard_linear_with_offset',
                   'kernel_with_offset', 'quan_linear']:
        list_of_std = [0.001]
        # list_of_lamb = [0.001]
        list_of_lamb = [1.0]
        list_of_option = ['distort_greedy', 'kl_triage']

    if dataset in ['quan_linear', 'quan_kernel']:
        list_of_std = [1.0]
        # list_of_lamb = [0.001]
        list_of_lamb = [1.0]
        list_of_option = ['kl_triage', 'distort_greedy']

    return list_of_option, list_of_std, list_of_lamb


class plot_triage_real:

    def __init__(self, list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option, flag_synthetic=None):
        self.list_of_K = list_of_K
        self.list_of_std = list_of_std
        self.list_of_lamb = list_of_lamb
        self.list_of_option = list_of_option
        self.list_of_test_option = list_of_test_option
        self.flag_synthetic = flag_synthetic
        self.threshold = 0.5

    def plot_subset(self, data_file, res_file, path, dataset, svm_type, option):
        data = load_data(data_file)
        res = load_data(res_file)
        list_of_K = [0.1, 0.2, 0.3, 0.4]
        # list_of_K = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]#[0.5]#[0.1, 0.2, 0.3, 0.4]
        lamb = self.list_of_lamb[0]
        # list_of_K = [0.1]
        for K in list_of_K:
            for std in self.list_of_std:
                # local_data = data[str(std)]
                local_data = {'X': data['X'], 'Y': data['Y'], 'c': data['c'][str(std)], 'dist_mat': data['dist_mat'],
                              'X_te': data['test']['X'], 'Y_te': data['test']['Y']}

                # print res.keys()
                local_res = res[str(std)][str(K)][str(lamb)][option]
                subset_human = local_res['subset']
                # subset_te_n, subset_machine_n, subset_te_r, subset_machine_r = self.get_test_subset(x_te=local_data['X_te'], x_tr=local_data['X'], subset=subset_human,
                #                                                                                     dist_mat=local_data['dist_mat'], test_method='NN')
                # print 'subset_human'
                # print K, subset_human.shape
                # w = local_res['w']
                n = local_data['X'].shape[0]
                subset_machine = np.array([i for i in range(n) if i not in subset_human])
                fig, ax = plt.subplots()
                fig.subplots_adjust(left=.1, bottom=.1, right=.97, top=.95)
                # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
                # color_list = bmap.mpl_colors
                color_list = get_color_list()
                x = local_data['X']
                # print x.shape
                # y = local_data['X'][:, 1]
                #
                machine_plus = np.array([idx for idx in subset_machine if local_data['Y'][idx] > 0])
                machine_minus = np.array([idx for idx in subset_machine if local_data['Y'][idx] <= 0])
                human_plus = np.array([idx for idx in subset_human if local_data['Y'][idx] > 0])
                human_minus = np.array([idx for idx in subset_human if local_data['Y'][idx] <= 0])

                # machine_plus = np.array([idx for idx in subset_machine_n if local_data['Y_te'][idx] > 0])
                # machine_minus = np.array([idx for idx in subset_machine_n if local_data['Y_te'][idx] <= 0])
                # human_plus = np.array([idx for idx in subset_te_n if local_data['Y_te'][idx] > 0])
                # human_minus = np.array([idx for idx in subset_te_n if local_data['Y_te'][idx] <= 0])

                X = local_data['X'][subset_machine, :]
                Y = local_data['Y'][subset_machine]
                from sklearn.svm import LinearSVC
                from sklearn import svm

                # model = svm.LinearSVC(C=0.01)
                reg_par = float(1) / (2.0 * lamb * subset_machine.shape[0])
                # reg_par=0.01
                if svm_type == 'hard_linear_with_offset':
                    model = svm.LinearSVC(C=1000)

                if svm_type == 'hard_linear_without_offset':
                    model = svm.LinearSVC(fit_intercept=False, C=1000)

                if svm_type == 'soft_linear_with_offset':
                    model = svm.LinearSVC(C=reg_par)

                if svm_type == 'soft_linear_without_offset':
                    model = svm.LinearSVC(fit_intercept=False, C=0.01)
                    # model = SVC(C=reg_par, kernel='linear')

                if svm_type == 'soft_kernel_with_offset':
                    # model = svm.SVC(kernel='poly', degree=2, C=reg_par)
                    model = svm.SVC(C=reg_par)

                if svm_type == 'soft_kernel_without_offset':
                    # model = SVC(C=reg_par,kernel='polynomial',degree=2)

                    plt.scatter(x[machine_plus, 0], x[machine_plus, 1], marker='+', s=90, linewidths=2,
                                c=color_list[0], zorder=10, cmap=plt.cm.Paired,
                                label=r'$\mathcal{V}$\textbackslash $\mathcal{S}^*$')
                    plt.scatter(x[machine_minus, 0], x[machine_minus, 1], marker='_', s=90, linewidths=2,
                                c=color_list[0], zorder=10, cmap=plt.cm.Paired)

                    # plt.scatter(local_data['X_te'][machine_plus, 0], local_data['X_te'][machine_plus, 1], marker='+', s=90, linewidths=2,
                    #             c=color_list[0], zorder=10, cmap=plt.cm.Paired,
                    #             label=r'$\mathcal{V}$\textbackslash $\mathcal{S}^*$')
                    # plt.scatter(local_data['X_te'][machine_minus, 0], local_data['X_te'][machine_minus, 1], marker='_', s=90, linewidths=2,
                    #             c=color_list[0], zorder=10, cmap=plt.cm.Paired)

                    if human_plus.shape[0] > 0:
                        plt.scatter(x[human_plus, 0], x[human_plus, 1], marker='+', s=90, linewidth=3,
                                    c=color_list[1], zorder=10, cmap=plt.cm.Paired,
                                    label=r'$\mathcal{S}^*$')
                        # plt.scatter(local_data['X_te'][human_plus, 0], local_data['X_te'][human_plus, 1], marker='+', s=90, linewidth=3,
                        #             c=color_list[1], zorder=10, cmap=plt.cm.Paired,
                        #             label=r'$\mathcal{S}^*$')

                    if human_minus.shape[0] > 0:
                        plt.scatter(x[human_minus, 0], x[human_minus, 1], marker='_', s=90, linewidth=3,
                                    c=color_list[1], zorder=10, cmap=plt.cm.Paired)
                        # plt.scatter(local_data['X_te'][human_minus, 0], local_data['X_te'][human_minus, 1], marker='_', s=90, linewidth=3,
                        #             c=color_list[1], zorder=10, cmap=plt.cm.Paired)
                    x_min = -10
                    x_max = 10
                    y_min = -10
                    y_max = 10

                    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
                    out = open('test_kernel_out.txt', 'w')
                    for xx, yy in zip(XX.ravel(), YY.ravel()):
                        line = '-1 '
                        line = line + '1:' + str(xx) + ' '
                        line = line + '2:' + str(yy) + '\n'
                        out.write(line)
                    out.close()
                    generate_svmlight('data/data_dict_kernel', subset_machine)
                    testreg_par = float(1) / (2.0 * lamb)
                    os.system(
                        '/home/nastaran/Downloads/bsvm-2.09/bsvm-train -s 1 -t 1 -d 2 -c 0.5 -g 0.5 data/data_dict_kernel_out.txt kernelmodel')
                    # w,sv = get_w('kernelmodel')
                    os.system(
                        '/home/nastaran/Downloads/bsvm-2.09/bsvm-predict test_kernel_out.txt kernelmodel kernel_result')
                    Z = get_f('kernel_result', XX.ravel().shape)

                    # def K(X1, X2):
                    #     gamma = 0.5
                    #     return (gamma * np.dot(X1, X2.T)) ** 2
                    #
                    # Z = np.dot(K(np.c_[XX.ravel(), YY.ravel()], sv), w.T)
                    Z = Z.reshape(XX.shape)
                    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
                    if dataset == 'Wsigmoid':
                        xlabel = '$[-7,7]$'
                        ax.set_ylim([-0.5, 1.5])
                        # x = -5

                    if dataset == 'Wgauss':
                        xlabel = '$[-1,1]$'
                        ax.set_ylim([0.17, 0.21])
                        # x = 3

                    if dataset in ['linear', 'kernel']:
                        xlabel = '$[-12,12]$'
                        # ax.set_ylim([10, 10])
                        # x = 3
                    import matplotlib.lines as mlines
                    green_dot = mlines.Line2D([], [], color=color_list[0], marker='o', linestyle='None',
                                              markersize=8, label=r'$\mathcal{V}$\textbackslash $\mathcal{S}$')
                    orange_dot = mlines.Line2D([], [], color=color_list[1], marker='o', linestyle='None',
                                               markersize=8, label=r'$\mathcal{S}$')
                    plt.legend(prop={'size': 19}, frameon=False, loc='upper left',
                               handlelength=0.1, handles=[green_dot, orange_dot])
                    # plt.xlabel(r'\textbf{Features} $x$ $\sim$ \textbf{Unif} ' + xlabel, fontsize=23)
                    # ax.set_ylabel(r'\textbf{Response} $(y)$', fontsize=23,
                    #               labelpad=5)

                    savepath = path + svm_type + '_' + str(K)[0] + '_' + str(K)[
                        2] + '_' + option
                    plt.savefig(savepath + '.pdf')
                    plt.savefig(savepath + '.png')
                    plt.close()
                    continue

                # model = svm.SVC(kernel='poly', degree=2, C=0.01)
                model.fit(X, Y)

                plt.scatter(x[machine_plus, 0], x[machine_plus, 1], marker='+', s=90, linewidths=2, c=color_list[0],
                            zorder=10, cmap=plt.cm.Paired,
                            label=r'$\mathcal{V}$\textbackslash $\mathcal{S}^*$')
                plt.scatter(x[machine_minus, 0], x[machine_minus, 1], marker='_', s=90, linewidths=2,
                            c=color_list[0], zorder=10, cmap=plt.cm.Paired)

                if human_plus.shape[0] > 0:
                    plt.scatter(x[human_plus, 0], x[human_plus, 1], marker='+', s=90, linewidth=3, c=color_list[1],
                                zorder=10, cmap=plt.cm.Paired,
                                label=r'$\mathcal{S}^*$')
                if human_minus.shape[0] > 0:
                    plt.scatter(x[human_minus, 0], x[human_minus, 1], marker='_', s=90, linewidth=3,
                                c=color_list[1], zorder=10, cmap=plt.cm.Paired)

                # plt.axis('tight')
                x_min = -10
                x_max = 10
                y_min = -10
                y_max = 10

                XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
                Z = model.decision_function(np.c_[XX.ravel(), YY.ravel()])
                # Z = model.predict(np.c_[XX.ravel(), YY.ravel()])

                # Put the result into a color plot
                Z = Z.reshape(XX.shape)
                # plt.figure(fignum, figsize=(4, 3))
                from matplotlib import colors

                plt.pcolormesh(XX, YY, Z > 0, cmap=colors.ListedColormap(['white', 'white']))
                plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                            levels=[-1, 0, 1])
                # plt.contourf(XX, YY, Z, cmap=plt.cm.coolwarm, alpha=0.8)
                #
                if dataset == 'Wsigmoid':
                    xlabel = '$[-7,7]$'
                    ax.set_ylim([-0.5, 1.5])
                    # x = -5

                if dataset == 'Wgauss':
                    xlabel = '$[-1,1]$'
                    ax.set_ylim([0.17, 0.21])
                    # x = 3

                if dataset in ['linear', 'kernel']:
                    xlabel = '$[-12,12]$'
                    # ax.set_ylim([10, 10])
                    # x = 3
                import matplotlib.lines as mlines
                green_dot = mlines.Line2D([], [], color=color_list[0], marker='o', linestyle='None',
                                          markersize=8, label=r'$\mathcal{V}$\textbackslash $\mathcal{S}$')
                orange_dot = mlines.Line2D([], [], color=color_list[1], marker='o', linestyle='None',
                                           markersize=8, label=r'$\mathcal{S}$')
                plt.legend(prop={'size': 19}, frameon=False, loc='upper left',
                           handlelength=0.1, handles=[green_dot, orange_dot])
                # plt.xlabel(r'\textbf{Features} $x$ $\sim$ \textbf{Unif} ' + xlabel, fontsize=23)
                # ax.set_ylabel(r'\textbf{Response} $(y)$', fontsize=23,
                #               labelpad=5)

                savepath = path + svm_type + '_' + str(K)[0] + '_' + str(K)[
                    2] + '_' + '_' + option  # + '_' + distort_greedy_type
                plt.savefig(savepath + '.pdf')
                plt.savefig(savepath + '.png')
                plt.close()

    def plot_subset_without_offset(self, data_file, res_file, list_of_std, list_of_lamb, path, dataset, svm_type,
                                   distort_greedy_type):
        data = load_data(data_file)
        res = load_data(res_file)
        # list_of_K = [0.2, 0.4, 0.6, 0.8]
        list_of_K = [0.1, 0.2, 0.3, 0.4]
        lamb = self.list_of_lamb[0]
        # list_of_K = [0.1]
        for K in list_of_K:
            for option in self.list_of_option:
                for std in list_of_std:
                    # local_data = data[str(std)]
                    local_data = {'X': data['X'], 'Y': data['Y'], 'c': data['c'][str(std)]}

                    # print res.keys()
                    local_res = res[str(std)][str(K)][str(lamb)][option]
                    subset_human = local_res['subset']
                    # w = local_res['w']
                    n = local_data['X'].shape[0]
                    subset_machine = np.array([i for i in range(n) if i not in subset_human])
                    fig, ax = plt.subplots()
                    fig.subplots_adjust(left=.1, bottom=.1, right=.97, top=.95)
                    # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
                    # color_list = bmap.mpl_colors
                    color_list = get_color_list()

                    x = local_data['X']
                    # print x.shape
                    # y = local_data['X'][:, 1]

                    machine_plus = np.array([idx for idx in subset_machine if local_data['Y'][idx] > 0])
                    machine_minus = np.array([idx for idx in subset_machine if local_data['Y'][idx] <= 0])
                    human_plus = np.array([idx for idx in subset_human if local_data['Y'][idx] > 0])
                    human_minus = np.array([idx for idx in subset_human if local_data['Y'][idx] <= 0])

                    X = local_data['X'][subset_machine, :]
                    Y = local_data['Y'][subset_machine]
                    from sklearn.svm import LinearSVC
                    from sklearn import svm

                    # model = svm.LinearSVC(C=0.01)
                    reg_par = float(1) / (2.0 * list_of_lamb[0] * subset_machine.shape[0])
                    if svm_type == 'soft_kernel_without_offset':
                        # model = SVC(C=reg_par,kernel='polynomial',degree=2)

                        plt.scatter(x[machine_plus, 0], x[machine_plus, 1], marker='+', s=90, linewidths=2,
                                    c=color_list[0], zorder=10, cmap=plt.cm.Paired,
                                    label=r'$\mathcal{V}$\textbackslash $\mathcal{S}^*$')
                        plt.scatter(x[machine_minus, 0], x[machine_minus, 1], marker='_', s=90, linewidths=2,
                                    c=color_list[0], zorder=10, cmap=plt.cm.Paired)

                        if human_plus.shape[0] > 0:
                            plt.scatter(x[human_plus, 0], x[human_plus, 1], marker='+', s=90, linewidth=3,
                                        c=color_list[1], zorder=10, cmap=plt.cm.Paired,
                                        label=r'$\mathcal{S}^*$')
                        if human_minus.shape[0] > 0:
                            plt.scatter(x[human_minus, 0], x[human_minus, 1], marker='_', s=90, linewidth=3,
                                        c=color_list[1], zorder=10, cmap=plt.cm.Paired)
                        x_min = -10
                        x_max = 10
                        y_min = -10
                        y_max = 10

                        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
                        out = open('test_kernel_out.txt', 'w')
                        for xx, yy in zip(XX.ravel(), YY.ravel()):
                            line = '-1 '
                            line = line + '1:' + str(xx) + ' '
                            line = line + '2:' + str(yy) + '\n'
                            out.write(line)
                        out.close()
                        generate_svmlight('data/data_dict_kernel', subset_machine)
                        testreg_par = float(1) / (2.0 * list_of_lamb[0])
                        os.system(
                            '/home/nastaran/Downloads/bsvm-2.09/bsvm-train -s 1 -t 1 -d 2 -c 0.5 -g 0.5 data/data_dict_kernel_out.txt kernelmodel')
                        # w,sv = get_w('kernelmodel')
                        os.system(
                            '/home/nastaran/Downloads/bsvm-2.09/bsvm-predict test_kernel_out.txt kernelmodel kernel_result')
                        Z = get_f('kernel_result', XX.ravel().shape)

                        # def K(X1, X2):
                        #     gamma = 0.5
                        #     return (gamma * np.dot(X1, X2.T)) ** 2
                        #
                        # Z = np.dot(K(np.c_[XX.ravel(), YY.ravel()], sv), w.T)
                        Z = Z.reshape(XX.shape)
                        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
                        if dataset == 'Wsigmoid':
                            xlabel = '$[-7,7]$'
                            ax.set_ylim([-0.5, 1.5])
                            # x = -5

                        if dataset == 'Wgauss':
                            xlabel = '$[-1,1]$'
                            ax.set_ylim([0.17, 0.21])
                            # x = 3

                        if dataset in ['linear', 'kernel']:
                            xlabel = '$[-12,12]$'
                            # ax.set_ylim([10, 10])
                            # x = 3
                        import matplotlib.lines as mlines
                        green_dot = mlines.Line2D([], [], color=color_list[0], marker='o', linestyle='None',
                                                  markersize=8, label=r'$\mathcal{V}$\textbackslash $\mathcal{S}$')
                        orange_dot = mlines.Line2D([], [], color=color_list[1], marker='o', linestyle='None',
                                                   markersize=8, label=r'$\mathcal{S}$')
                        plt.legend(prop={'size': 19}, frameon=False, loc='upper left',
                                   handlelength=0.1, handles=[green_dot, orange_dot])
                        # plt.xlabel(r'\textbf{Features} $x$ $\sim$ \textbf{Unif} ' + xlabel, fontsize=23)
                        # ax.set_ylabel(r'\textbf{Response} $(y)$', fontsize=23,
                        #               labelpad=5)

                        savepath = path + svm_type + '_' + str(K)[0] + '_' + str(K)[
                            2] + '_' + option  # + '_' + distort_greedy_type
                        plt.savefig(savepath + '.pdf')
                        plt.savefig(savepath + '.png')
                        plt.close()

    def U_get_avg_error_vary_K(self, res_file, test_method, dataset, path):
        res = load_data(res_file)
        savepath = path + dataset + '_' + test_method

        real = ['Umessidor', 'Ustare5', 'Ustare11']
        synthetic = ['Usigmoid', 'Ugauss']

        assert dataset in real or dataset in synthetic

        if dataset in real:
            multtext = r'$\times 10^{-1}$'
            labeltext = r'$\rho_c = $'
            mult = 10

        if dataset in synthetic:
            multtext = r'$\times 10^{-3}$'
            labeltext = r'$\sigma_2 = $'
            mult = 1000

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        # color_list = bmap.mpl_colors
        color_list = get_color_list()
        plt.figtext(0.115, 0.96, multtext, fontsize=17)

        for idx, std in enumerate(self.list_of_std):
            for lamb in self.list_of_lamb:
                for option, ind in zip(self.list_of_option, range(len(self.list_of_option))):
                    err_K_tr = []
                    err_K_te = []

                    for K in self.list_of_K:
                        err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
                        err_K_te.append(
                            res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])

                    err_K_te_mult = [mult * err for err in err_K_te]  # for synthetic
                    ax.plot((err_K_te_mult), label=labeltext + str(std), linewidth=3, marker='o',
                            markersize=10, color=color_list[idx])

        ax.legend(prop={'size': 18}, frameon=False, handlelength=0.2, loc='best')
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25, labelpad=3)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png')
        plt.close()

    def U_get_avg_error_vary_testmethod(self, res_file, dataset, path, std):

        res = load_data(res_file)
        real = ['Umessidor', 'Ustare5', 'Ustare11']
        synthetic = ['Usigmoid', 'Ugauss']

        assert dataset in real or dataset in synthetic

        savepath = path + dataset + '_' + str(std) + '_vary_testmethod'
        if dataset in real:
            multtext = r'$\times 10^{-1}$'
            mult = 10

        if dataset in synthetic:
            multtext = r'$\times 10^{-3}$'
            mult = 1000

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        # color_list = bmap.mpl_colors
        color_list = get_color_list()
        plt.figtext(0.115, 0.96, multtext, fontsize=17)

        for lamb in self.list_of_lamb:
            for option, ind in zip(self.list_of_option, range(len(self.list_of_option))):
                for idx, test_method in enumerate(self.list_of_test_option):
                    label_map = {'MLP': 'Multilayer perceptron', 'LR': 'Logistic regression', 'NN': 'NN neighbor'}
                    err_K_tr = []
                    err_K_te = []

                    for K in self.list_of_K:
                        err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
                        err_K_te.append(
                            res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])

                    err_K_te_mult = [mult * err for err in err_K_te]  # for synthetic
                    ax.plot((err_K_te_mult), label=label_map[test_method], linewidth=3, marker='o',
                            markersize=10, color=color_list[idx])

        ax.legend(prop={'size': 18}, frameon=False, handlelength=0.2, loc='best')
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25, labelpad=3)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png')
        plt.close()

    def get_avg_error_vary_K(self, res_file,data_file, image_path, file_name, test_method):
        res = load_data(res_file)

        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}

                for option in self.list_of_option:
                    err_K_te = []
                    for K in self.list_of_K:
                        err_K_te.append(
                            res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])

                    plot_obj[option] = {'test': err_K_te}

                new_image_path = image_path
                self.plot_err_vs_K(data_file, new_image_path, plot_obj, file_name, test_method, std,lamb)

    def get_avg_error_vary_testmethod(self, res_file, image_path, file_name, option):
        res = load_data(res_file)

        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}

                for test_method in self.list_of_test_option:
                    err_K_te = []

                    for K in self.list_of_K:
                        err_K_te.append(
                            res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])

                    plot_obj[test_method] = {'test': err_K_te}

                self.plot_err_vs_testmethod(image_path, plot_obj, file_name)

    def plot_err_vs_testmethod(self, image_file, plot_obj, file_name):
        savepath = image_file + file_name + '_vary_testmethod'
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        # color_list = bmap.mpl_colors
        color_list = get_color_list()
        key = 'test'
        synthetic = ['sigmoid', 'gauss']
        mult = 10
        multtext = r'$\times 10^{-1}$'

        if file_name in synthetic:
            mult = 1000
            multtext = r'$\times 10^{-3}$'

        for idx, option in enumerate(plot_obj.keys()):
            err = [x * mult for x in plot_obj[option][key]]
            label_map = {'MLP': 'Multilayer perceptron', 'LR': 'Logistic regression', 'NN': 'NN neighbor'}
            plt.plot(err, label=label_map[option], linewidth=3, marker='o',
                     markersize=10, color=color_list[idx])

        plt.figtext(0.115, 0.95, multtext, fontsize=17)
        plt.legend(prop={'size': 18}, frameon=False, handlelength=0.2)
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25, labelpad=2)

        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png')
        # save(plot_obj, image_file)
        plt.close()

    def get_machine_pred(self,X_tr,Y_tr, X_te):
        from sklearn.svm import SVC
        from sklearn import svm
        lamb = self.list_of_lamb[0]
        reg_par = float(1) / (2.0 * lamb * X_tr.shape[0])
        model = SVC(C=30,gamma=0.0001)
        # model = svm.LinearSVC(C=reg_par)
        # model = SVC(kernel='poly',C=reg_par,degree=2)
        Y = self.get_labels(Y_tr)
        model.fit(X_tr,Y)
        y_pred = model.predict(X_te)
        return y_pred

    def plot_err_vs_K(self, data_file, image_file, plot_obj, file_name, test_method, std,lamb):

        savepath = image_file + file_name + '_' + test_method
        print savepath
        data = load_data(data_file)
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        # color_list = bmap.mpl_colors
        color_list = get_color_list()
        synthetic = ['sigmoid', 'gauss']
        mult = 10
        multtext = r'$\times 10^{-1}$'

        if file_name in synthetic:
            mult = 1000
            multtext = r'$\times 10^{-3}$'

        key = 'test'
        for idx, option in enumerate(plot_obj.keys()):
            err = [x * mult for x in plot_obj[option][key]]
            label_map = {'kl_triage': 'Triage', 'distort_greedy': 'DG', 'stochastic_distort_greedy': 'Stochastic DG',
                         'greedy': 'Greedy', 'diff_submod': 'DS', 'RLSR_Reg': 'CRR'}

            ax.plot(err, label=label_map[option], linewidth=3, marker='o',
                    markersize=10, color=color_list[idx])

        human_pred = data['test']['y_h'][str(std)]
        machine_pred = self.get_machine_pred(data['X'], data['Y'],data['test']['X'])
        true_pred = self.get_labels(data['test']['Y'])
        human_error = np.sum(human_pred!=true_pred)*mult/float(true_pred.shape[0])
        machine_error = np.sum(machine_pred!=true_pred)*mult/float(true_pred.shape[0])
        print machine_error
        # print human_error
        human_obj = []
        machine_obj = []
        for K in self.list_of_K:
            human_obj.append(human_error)
            machine_obj.append(machine_error)

        ax.plot(machine_obj, linewidth=3, linestyle='--', label='Full Automation',
                color=color_list[5])

        ax.plot(human_obj, linewidth=3, linestyle='--', label='No Automation',
                color=color_list[7])
        plt.figtext(0.115, 0.95, multtext, fontsize=17)
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 2, 1, 3]#, 4]  # [4, 0, 1, 2, 3]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 13}, frameon=False,
                  handlelength=0.9)
        # ax.legend(handles=[green_dot,orange_dot,dashed_line1,dashed_line2], prop={'size': 17}, frameon=False,
        #                     handlelength=0.5)
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{Misclassify Error}', fontsize=20, labelpad=2)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '_' + str(std) + '_' + str(lamb) + '.pdf')
        plt.savefig(savepath + '_' + str(std) + '_' + str(lamb) + '.png')
        # save(plot_obj, image_file)
        plt.close()

    def get_NN_human(self, dist, tr_human_ind):
        n_tr = dist.shape[0]
        human_dist = float('inf')
        machine_dist = float('inf')

        for d, tr_ind in zip(dist, range(n_tr)):
            if tr_ind in tr_human_ind:
                if d < human_dist:
                    human_dist = d

            else:
                if d < machine_dist:
                    machine_dist = d

        return human_dist - machine_dist

    def get_test_subset(self, x_te, x_tr, subset, dist_mat, test_method):
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn import svm

        y = np.zeros(x_tr.shape[0], dtype='uint')
        y[subset] = 1  # human label = 1
        if subset.shape[0] == 0:
            subset_te_r = np.array([])
            subset_machine_r = np.arange(x_te.shape[0])
            subset_te_n = np.array([])
            subset_machine_n = np.arange(x_te.shape[0])
            return subset_te_n, subset_machine_n, subset_te_r, subset_machine_r

        if test_method == 'MLP':
            model = MLPClassifier(max_iter=400)
        if test_method == 'LR':
            model = LogisticRegression(C=0.01)
            # from sklearn.tree import DecisionTreeClassifier
            # model = DecisionTreeClassifier(max_depth=5)
            # model = svm.SVC(gamma=2,C=1)
            # model = MLPClassifier(max_iter=500)
        if test_method == 'NN':
            # from sklearn.neighbors import KNeighborsClassifier
            # model = KNeighborsClassifier()
            n, tr_n = dist_mat.shape
            no_human = int((subset.shape[0] * n) / float(tr_n))
            diff_arr = [self.get_NN_human(dist, subset) for dist in dist_mat]
            indices = np.argsort(np.array(diff_arr))
            subset_te_r = indices[:no_human]
            subset_machine_r = indices[no_human:]
            subset_te_n = np.array([int(i) for i in range(len(diff_arr)) if diff_arr[i] < 0])
            subset_machine_n = np.array([int(i) for i in range(len(diff_arr)) if i not in subset_te_n])
            # print subset.shape, subset_te_n.shape
            return subset_te_n, subset_machine_n, subset_te_r, subset_machine_r

        # print y
        model.fit(x_tr, y)
        y_pred = model.predict(x_te)
        subset_te_r = []
        subset_machine_r = []
        print 'x_tr o x_te'
        print x_tr.shape, x_te.shape, np.sum(y), subset.shape, np.sum(y_pred)

        for idx, label in enumerate(y_pred):
            if label == 1:
                subset_te_r.append(idx)
            else:
                subset_machine_r.append(idx)

        subset_machine_r = np.array(subset_machine_r)
        subset_te_r = np.array(subset_te_r)
        subset_te_n = np.array([int(i) for i in range(len(y_pred)) if y_pred[i] == 1])
        subset_machine_n = np.array([int(i) for i in range(len(y_pred)) if i not in subset_te_n])
        # print subset.shape, subset_te_n.shape
        # assert (subset_te_n.all() == subset_te_r.all())
        return subset_te_n, subset_machine_n, subset_te_r, subset_machine_r

    def plot_roc(self, res_file, data_file, option, dataset, test_method):

        for std in self.list_of_std:
            from sklearn.metrics import roc_auc_score
            from sklearn.metrics import roc_curve, auc
            res = load_data(res_file)
            # data = load_data(data_file)
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=.15, bottom=.16, right=.97, top=.93)
            # bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
            # color_list = bmap.mpl_colors
            color_list = get_color_list()
            for idx, K in enumerate(self.list_of_K):
                if idx % 2:
                    continue
                for lamb in self.list_of_lamb:
                    # subset = res[str(std)][str(K)][str(lamb)][option]['subset']
                    # subset_c = np.array([int(i) for i in range(self.n) if i not in subset])
                    y_score = res[str(std)][str(K)][str(lamb)][option]['y_score']
                    y_test = res[str(std)][str(K)][str(lamb)][option]['y_te']

                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()

                    # fpr, tpr, _ = roc_curve(y_test, y_score)
                    # roc_auc = auc(fpr, tpr)

                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                    # plt.figure()
                    lw = 2

                    plt.plot(fpr['micro'], tpr['micro'], color=color_list[idx / 2],
                             lw=lw, label=str(K) + r' (area = %0.2f)' % roc_auc['micro'])
                    plt.plot([0, 1], [0, 1], color=color_list[6], lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel(r'\textbf{False Positive Rate}', fontsize=22)
            plt.ylabel(r'\textbf{True Positive Rate}', fontsize=22)
            # plt.title(r'Receiver Operating Characteristic curve')
            plt.legend(loc="lower right", prop={'size': 16}, frameon=False)
            plt.savefig('plots/' + 'auc_' + option + '_' + dataset + '_' + test_method + '_' + str(std) + '.png')
            plt.savefig('plots/' + 'auc_' + option + '_' + dataset + '_' + test_method + '_' + str(std) + '.pdf')
            plt.close()

    def plot_test_errors(self, res_file, data_file, option, file_name, test_method):
        res = load_data(res_file)
        data = load_data(data_file)
        for std in self.list_of_std:
            for K in self.list_of_K:
                for lamb in self.list_of_lamb:
                    res_obj = res[str(std)][str(K)][str(lamb)][option]
                    human_error = data['test']['c'][str(std)]
                    X_te = data['test']['X']
                    Y_te = data['test']['Y']
                    Y_discrete = self.get_labels(Y_te)
                    h_machine = res_obj['y_score']
                    subset_human = res_obj['test_res'][test_method][test_method]['human_ind']
                    subset_machine = res_obj['test_res'][test_method][test_method]['machine_ind']
                    from sklearn.metrics import hinge_loss
                    # print Y_te.shape,h_machine.shape
                    print h_machine
                    machine_error = 1 - (Y_discrete*h_machine)
                    np.clip(machine_error, 0, None, out=machine_error)
                    # plt.subplot(2,1,1)
                    human_barlist = plt.bar(range(X_te.shape[0]),human_error-machine_error)
                    for idx,bar in enumerate(human_barlist):
                        if idx in subset_human:
                            bar.set_color(get_color_list()[3])
                    # plt.subplot(2,1,2)
                    # machine_barlist = plt.bar(range(X_te.shape[0]),machine_error)
                    # for idx,bar in enumerate(machine_barlist):
                    #     if idx in subset_human:
                    #         bar.set_color(get_color_list()[1])


                    plt.show()


    def classification_get_test_error(self, res_obj, dist_mat, test_method, x_tr, y_tr, x, y, svm_type, h_test,
                                      y_h=None, c=None):
        # subset_size = int(K * x_tr.shape[0])

        subset = res_obj['subset']
        print subset.shape
        subset_c = np.array([int(i) for i in range(self.n) if i not in subset])

        n, tr_n = dist_mat.shape

        subset_te_n, subset_machine_n, subset_te_r, subset_machine_r = self.get_test_subset(x_te=x, x_tr=x_tr,
                                                                                            subset=subset,
                                                                                            dist_mat=dist_mat,
                                                                                            test_method=test_method)
        X_tr = x_tr[subset_c]
        Y_tr = y_tr[subset_c]
        from sklearn import svm
        reg_par = float(1) / (2.0 * self.list_of_lamb[0]*subset_c.shape[0])
        if svm_type == 'hard_linear_with_offset':
            model = svm.LinearSVC(C=1000)

        if svm_type == 'hard_linear_without_offset':
            model = svm.LinearSVC(fit_intercept=False, C=1000)

        if svm_type == 'soft_linear_with_offset':
            model = svm.LinearSVC(C=reg_par)

        if svm_type == 'soft_linear_without_offset':
            model = svm.LinearSVC(fit_intercept=False, C=reg_par)
            # model = SVC(C=reg_par, kernel='linear')

        if svm_type == 'soft_kernel_with_offset':
            # model = svm.SVC(kernel='poly', degree=2, C=reg_par)
            model = svm.SVC(C=30,gamma=0.0001)
            # model = svm.SVC(kernel='poly',C=reg_par,degree=2)
            # from sklearn import svm
            # model = svm.LinearSVC(C=reg_par)

        # model = svm.SVC(C=0.01)
        model.fit(X_tr, Y_tr)
        y_pred = model.predict(x)
        print y_pred
        y_score = model.decision_function(x)
        err_m = (y_pred != y)
        err_h = (y_h != y)
        # print y_score

        if subset_te_r.size == 0:
            error_r = err_m.sum() / float(n)
        else:
            error_r = (err_h[subset_te_r].sum() + err_m[subset_machine_r].sum()) / float(n)

        if subset_te_n.size == 0:
            error_n = err_m.sum() / float(n)
        else:
            error_n = (err_h[subset_te_r].sum() + err_m[subset_machine_r].sum()) / float(n)

        error_n = {'error': error_n, 'human_ind': subset_te_n, 'machine_ind': subset_machine_n}
        error_r = {'error': error_r, 'human_ind': subset_te_r, 'machine_ind': subset_machine_r}

        return error_n, error_r, y_score


    def plot_test_allocation(self, train_obj, test_obj, plot_file_path):
        x = train_obj['human']['x']
        y = train_obj['human']['y']
        plt.scatter(x, y, c='blue', label='train human')

        x = train_obj['machine']['x']
        y = train_obj['machine']['y']
        plt.scatter(x, y, c='green', label='train machine')

        x = test_obj['machine']['x'][:, 0].flatten()
        y = test_obj['machine']['y']
        plt.scatter(x, y, c='yellow', label='test machine')

        x = test_obj['human']['x'][:, 0].flatten()
        y = test_obj['human']['y']
        plt.scatter(x, y, c='red', label='test human')

        plt.legend()
        plt.grid()
        plt.xlabel('<-----------x------------->')
        plt.ylabel('<-----------y------------->')
        plt.savefig(plot_file_path, dpi=600, bbox_inches='tight')
        plt.close()

    def get_train_error(self, plt_obj, x, y, y_h=None, c=None):
        subset = plt_obj['subset']
        # print subset.shape
        n = y.shape[0]

        # if y_h == None:
        #     err_h = c
        #
        # else:
        err_h = (y != y_h)
        # err_h = (y_h - y) ** 2

        # y_m = x.dot(w)
        # err_m = (y_m - y) ** 2
        # error = (err_h[subset].sum() + err_m.sum() - err_m[subset].sum()) / float(n)

        subset_c = np.array([int(i) for i in range(self.n) if i not in subset])
        X = x[subset_c]
        Y = y[subset_c]

        from sklearn import svm
        from sklearn.metrics import hinge_loss
        reg_par = float(1) / (2.0 * self.list_of_lamb[0] * subset_c.shape[0])
        # model = svm.LinearSVC(C=reg_par)
        model = svm.SVC(C=reg_par)
        # model = svm.SVC(kernel='linear', C=reg_par)
        model.fit(X, Y)
        # y_pred = model.decision_function(X)
        y_pred = model.predict(X)
        # hinge_machine_loss = hinge_loss(Y, y_pred)
        # hinge_machine_loss *= y_pred.shape[0]
        err_m = (y_pred != Y)
        error = (err_h[subset].sum() + err_m.sum()) / float(n)

        return {'error': error}

    def get_labels(self, cont_y):
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
        print cntplus, cntminus
        return y

    def compute_result(self, res_file, data_file, option, test_method, svm_type):
        data = load_data(data_file)
        res = load_data(res_file)

        for std, i0 in zip(self.list_of_std, range(len(self.list_of_std))):
            for K, i1 in zip(self.list_of_K, range(len(self.list_of_K))):
                for lamb, i2 in zip(self.list_of_lamb, range(len(self.list_of_lamb))):

                    if option in res[str(std)][str(K)][str(lamb)]:
                        res_obj = res[str(std)][str(K)][str(lamb)][option]
                        # data = data[str(std)]
                        Y_tr = self.get_labels(data['Y'])
                        Y_te = self.get_labels(data['test']['Y'])
                        train_res = self.get_train_error(res_obj, data['X'], Y_tr, y_h=data['y_h'][str(std)],
                                                         c=data['c'][str(std)])

                        # if test_method == 'NN':
                        #     test_res_n, test_res_r = self.get_test_error(res_obj=res_obj, dist_mat=data['dist_mat'], x_tr=data['X'], y_tr=data['Y'], x=data['test']['X'],
                        #                                                  y=data['test']['Y'], y_h=None,
                        #                                                  c=data['test']['c'][str(std)])
                        # else:

                        test_res_n, test_res_r, y_score = self.classification_get_test_error(res_obj=res_obj,
                                                                                                   dist_mat=data[
                                                                                                       'dist_mat'],
                                                                                                   test_method=test_method,
                                                                                                   x_tr=data['X'],
                                                                                                   y_tr=Y_tr,
                                                                                                   x=data['test']['X'],
                                                                                                   y=Y_te, h_test=
                                                                                                   data['test']['h'][
                                                                                                       str(std)],
                                                                                                   y_h=
                                                                                                   data['test']['y_h'][
                                                                                                       str(std)],
                                                                                                   c=data['test']['c'][
                                                                                                       str(std)],
                                                                                                   svm_type=svm_type)

                        if 'test_res' not in res[str(std)][str(K)][str(lamb)][option]:
                            res[str(std)][str(K)][str(lamb)][option]['test_res'] = {}

                        if 'y_score' not in res[str(std)][str(K)][str(lamb)][option]:
                            res[str(std)][str(K)][str(lamb)][option]['y_score'] = {}

                        # if 'y_te' not in res[str(std)][str(K)][str(lamb)][option]:
                        #     res[str(std)][str(K)][str(lamb)][option]['y_te'] = {}

                        res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method] = {'ranking': test_res_r,
                                                                                             test_method: test_res_n}
                        res[str(std)][str(K)][str(lamb)][option]['train_res'] = train_res
                        res[str(std)][str(K)][str(lamb)][option]['y_score'] = y_score
                        # res[str(std)][str(K)][str(lamb)][option]['y_te'] = y_te

        save(res, res_file)

    def set_n(self, n):
        self.n = n


def main():
    latexify()
    list_of_test_option =['LR', 'MLP', 'NN']  # ,'MLP']

    my_parser = parser.opts_parser()
    args = my_parser.parse_args()
    args = vars(args)
    list_of_file_names = [args['dataset']]
    svm_type = args['svm_type']

    image_path = 'plots/'
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    for file_name in list_of_file_names:
        print 'plotting ' + file_name
        list_of_option, list_of_std, list_of_lamb = parse_command_line_input(file_name)
        list_of_K = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]#, 0.45,0.5]#,0.55,0.6]#,0.65,0.7,0.75,0.8]#,0.85, 0.9]

        data_file = 'data/data_dict_' + file_name
        res_file = 'Results/' + file_name + '_' + svm_type + '_res'

        obj = plot_triage_real(list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option)
        if file_name in ['quan_kernel', 'linear_with_offset', 'kernel', 'Wgauss', 'Wsigmoid', 'linear', 'hard_linear',
                         'hard_linear_with_offset', 'kernel_with_offset']:
            for option in list_of_option:
                if svm_type != 'soft_kernel_without_offset':

                    # if option not in ['diff_submod', 'RLSR', 'RLSR_Reg']:
                    #     unified_K = 0.5
                    #     obj.split_res_over_K(data_file, res_file, unified_K, option)
                    obj.plot_subset(data_file=data_file, res_file=res_file,
                                    path=image_path, dataset=file_name, svm_type=svm_type,
                                     option=option)
                else:
                    obj.plot_subset_without_offset(data_file, res_file, list_of_std, list_of_lamb, image_path,
                                                   file_name, svm_type)

        if file_name in ['kernel', 'linear_with_offset', 'quan_kernel', 'messidor']:
            # print load_data(data_file).keys()
            obj.set_n(load_data(data_file)['X'].shape[0])
            for idx, test_method in enumerate(list_of_test_option):
                for option in list_of_option:
                    if option not in ['diff_submod', 'RLSR', 'RLSR_Reg']:
                        # unified_K = 0.5
                        # obj.split_res_over_K(data_file, res_file, unified_K, option)
                        # print 'here'
                    # print option
                        obj.compute_result(res_file, data_file, option, test_method, svm_type)
                    # obj.plot_roc(res_file, data_file, option, file_name, test_method)
                # obj.plot_test_errors(res_file,data_file,'stochastic_distort_greedy',file_name,test_method)

                if file_name.startswith('U'):
                    obj.U_get_avg_error_vary_K(res_file, test_method, file_name, image_path)
                else:
                    savepath = image_path + '/' + file_name + '/'
                    if not os.path.exists(savepath):
                        os.mkdir(savepath)
                    obj.get_avg_error_vary_K(res_file, data_file, savepath,
                                             file_name, test_method)
            # if not file_name.startswith('U'):
            #     obj.get_avg_error_vary_testmethod(res_file, image_path, file_name, 'greedy')


if __name__ == "__main__":
    main()
