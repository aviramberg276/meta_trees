import argparse

import torch
from model_utils.evaluation import _prepare_x_y
from model_utils.utils import to_x_y, DIST_HARD
from model_utils.visualize import render_tree
from models.tree_model_dyn import get_regression_tree
from run_utils import get_data
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model path (src/models/<data_type>/<weights_filename>)')
    parser.add_argument('--result_dir', help='dir for the output trees')
    parser.add_argument('--data', choices=['ml_100k', 'ml_1m', 'jester'], default='ml_1m')
    args = parser.parse_args()
    return args


def run(model_path, result_dir, data):
    print("Load model: {0}".format(model_path))
    model = get_regression_tree(23, 1.0, 5.0, 512, 6, DIST_HARD, 1.0)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    print("Process {0} dataset".format(data))
    train, test = get_data(data)
    group_parameter = 'user_id'
    target = 'rating'
    x_train, y_train = to_x_y(train, group_parameter, target)
    columns = x_train.columns
    preprocessor = StandardScaler()
    preprocessor.fit(x_train.astype('float'))
    model.set_dist_func(DIST_HARD)
    groups = train.groupby(group_parameter).size().sort_values()
    isDummy = [False, True, True, True, True, True, True, True, True, True,
               True, True, True, True, True, True, True, True, True, False,
               True, False, False]
    for group, size in groups[groups > 50].iteritems():
        group_train = train[train[group_parameter] == group]
        x_train, y_train = _prepare_x_y(group_train, group_parameter, target, preprocessor, False)
        y_train = y_train.reshape(1, -1, 1)
        g = render_tree(model, x_train, y_train, preprocessor, range(1), columns, isDummy)
        g.render('{2}/user_id-{0}_size-{1}_hard'.format(group, size, result_dir))



if __name__ == '__main__':
    args = parse_args()
    run(args.model_path, args.result_dir, args.data)
