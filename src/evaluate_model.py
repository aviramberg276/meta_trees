import argparse

import torch
from model_utils.evaluation import get_regression_performance
from model_utils.utils import to_x_y, DIST_SOFT, DIST_HARD, GUMBEL_HARD
from models.tree_model_dyn import get_regression_tree
from run_utils import get_data
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model path (src/models/<data_type>/<weights_filename>)')
    parser.add_argument('--data', choices=['ml_100k', 'ml_1m', 'jester'], default='ml_1m')
    args = parser.parse_args()
    return args

def run(model_path, data):
    print("Load model: {0}".format(model_path))
    model = get_regression_tree(23, 1.0, 5.0, 512, 6, DIST_HARD, 1.0)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    print("Process {0} dataset".format(data))
    train, test = get_data(data)
    group_parameter = 'user_id'
    target = 'rating'
    x_train, y_train = to_x_y(train, group_parameter, target)
    preprocessor = StandardScaler()
    preprocessor.fit(x_train.astype('float'))
    model.set_dist_func(DIST_SOFT)
    rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    print('RMSE {0}, MAE {1} (soft, soft)'.format(rmse, mae))
    model.set_dist_func(DIST_HARD)
    rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    print('RMSE {0}, MAE {1} (hard, soft)'.format(rmse, mae))
    model.set_dist_func(GUMBEL_HARD)
    rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    print('RMSE {0}, MAE {1} (gumbel, soft)'.format(rmse, mae))


if __name__ == '__main__':
    args = parse_args()
    run(args.model_path, args.data)


