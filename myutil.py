import os
import pickle
import numpy as np


def save(obj, output_file):
    with open(output_file + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def write_txt_file(data, file_name):
    with open(file_name, 'w') as f:
        for line in data:
            f.write(" ".join(map(str, line)) + '\n')


def ma(y, window):
    avg_mask = np.ones(window) / window
    y_ma = np.convolve(y, avg_mask, 'same')
    y_ma[0] = y[0]
    y_ma[-1] = y[-1]
    return y_ma


def load_data(input_file, flag=None):
    if flag == 'ifexists':
        if not os.path.isfile(input_file + '.pkl'):
            # print 'not found', input_file
            return {}
    # print 'found'
    with open(input_file + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def generate_svmlight(file_name, subset, std=1):
    file = open(file_name+'.pkl')
    # print file
    file = pickle.load(file)
    # X = file[str(std)]['X']
    # Y = file[str(std)]['Y']
    X = file['X']
    Y = file['Y']

    X = X[subset]
    Y = Y[subset]

    def svm_parse(X, Y):
        def _convert(t):
            """Convert feature and value to appropriate types"""
            return (int(t[0]), float(t[1]))

        for x, y in zip(X, Y):
            # line = line.strip()
            # if not line.startswith('#'):
            # line = line.split('#')[0].strip() # remove any trailing comment
            # data = line.split()
            target = float(y)
            features = [_convert([i, feature]) for i, feature in enumerate(x)]
            yield (target, features)

    training_data = list(svm_parse(X, Y))

    out = open(file_name + '_out.txt', 'w')
    # print out
    for data in training_data:
        line = str(int(data[0])) + ' '
        line = line + str(data[1][0][0] + 1) + ':' + str(data[1][0][1]) + ' '
        line = line + str(data[1][1][0] + 1) + ':' + str(data[1][1][1]) + '\n'
        out.write(line)
    out.close()

def get_w(file_name):
    file = open(file_name)
    for line in file.readlines():
        if line.startswith('total_sv'):
            num_sv = int(line.split(' ')[1][:-1])
            break

    coef = np.zeros((num_sv),dtype='float')
    sv = np.zeros((num_sv, 2),dtype='float')

    f = False
    file = open(file_name)
    idx = 0

    for line in file.readlines():
        if (not f) and line.startswith('SV'):
            f = True
            continue

        if f:
            splited = line.split(' ')
            c = float(splited[0])
            valx = float(splited[1].split(':')[1])
            valy = float(splited[2].split(':')[1])
            # print coef,valx,valy
            coef[idx] = c
            sv[idx][1] = valx
            sv[idx][0] = valy
            idx += 1
        # line_x[idx] =

    # print coef
    # print sv
    w = np.reshape(coef, (1, coef.shape[0]))
    # w = np.dot(coef, sv)
    return w,sv

def get_f(file_name,shape):
    file = open(file_name)
    Z = np.zeros(shape)
    index = 0
    for z in file.readlines():
        Z[index] = float(z)
        index += 1
    return Z

def get_hinge_loss(file_name):
    file = open(file_name)
    for idx,line in enumerate(file.readlines()):
        if idx==0:
            hinge_loss = line[:-1]
        else:
            reg_loss = line[:-1]

    return float(hinge_loss),float(reg_loss)

def get_train_f(file_name,shape):
    file = open(file_name)
    Z = np.zeros(shape)
    index = 0
    for z in file.readlines():
        Z[index] = float(z)
        index += 1
    return Z

def get_color_list():
    color_list = [(0.4, 0.7607843137254902, 0.6470588235294118),
             (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
             (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
             (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
             (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
             (1.0, 0.8509803921568627, 0.1843137254901961),
             (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
             (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)]

    return color_list

