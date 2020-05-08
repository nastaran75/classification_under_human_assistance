from argparse import ArgumentParser


def opts_parser():
    parser = ArgumentParser()
    parser.add_argument(
            '--dataset', type=str, default='linear',
            help='dataset name (default: %(default)s)')

    parser.add_argument(
        '--svm_type', type=str, default='soft_linear_without_offset',
        help='svm_type (default: %(default)s)')


    # parser.add_argument(
    #         '--is_hard', action='store_true', default = False,
    #         help='train hard-margin svm instead of soft-margin svm (False by default)')
    #
    # parser.add_argument(
    #         '--with_offset', action='store_true', default = 'False',
    #         help='train svm with offset (False by default)')
    #
    # parser.add_argument(
    #     '--kernel_type', type=str, default='linear',
    #     help='linear svm or kernel svm (default: linear)')

    parser.add_argument(
        '--stochastic', action='store_true', default = False,
        help='train stochastic-distorted-greedy (False by default)')

    return parser