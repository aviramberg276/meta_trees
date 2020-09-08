import argparse

import torch
from sklearn.preprocessing import StandardScaler

from src.model_utils.evaluation import get_regression_performance
from src.model_utils.utils import to_x_y, DIST_SOFT, DIST_HARD, GUMBEL_HARD
from src.run_utils import get_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model path', default='')
    parser.add_argument('--data', choices=['ml_100k', 'ml_1m', 'jester'], default='ml_100k')
    args = parser.parse_args()
    return args

def run(model_path, data):
    print("Load model: {0}".format(model_path))
    model = torch.load(model_path)
    model.eval()
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
    print('RMSE {0}, MAE {1} (hard, soft)'.format(rmse, mae))


if __name__ == '__main__':
    # args = parse_args()
    # run(args.model_path, args.data)
    run("./models/model.pkl", "ml_100k")
