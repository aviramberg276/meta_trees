import torch
from sklearn.preprocessing import StandardScaler

from src.model_utils.evaluation import get_regression_performance
from src.model_utils.utils import DIST_HARD, DIST_SOFT, GUMBEL_HARD, to_x_y
from src.models.tree_model_dyn import get_regression_tree as get_regression_tree_dyn
from src.run_utils import get_dist_func, get_data

if __name__ == '__main__':
    model = get_regression_tree_dyn(23, 1.0, 5.0, 512, 5, get_dist_func("hard"), 1.0)
    model_file_name = "./models/ml_1m/height(0)_rdim(512)_temp(1.0)_rsparse(0.1)_dist(hard)_batch_size(256)_lr(0.0003)"

    model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')))
    torch.save(model, "./height(0)_rdim(512)_temp(1.0)_rsparse(0.1)_dist(hard)_batch_size(256)_lr(0.0003).pkl")

    model.eval()

    # train, test = get_data('ml_1m')
    # group_parameter = 'user_id'
    # target = 'rating'
    # x_train, y_train = to_x_y(train, group_parameter, target)
    # columns = x_train.columns
    # preprocessor = StandardScaler()
    # preprocessor.fit(x_train.astype('float'))
    #
    # model.set_dist_func(DIST_SOFT)
    # rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    # print('RMSE {0}, MAE {1} (soft, soft)'.format(rmse, mae))
    # model.set_dist_func(DIST_HARD)
    # rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    # print('RMSE {0}, MAE {1} (hard, soft)'.format(rmse, mae))
    # model.set_dist_func(GUMBEL_HARD)
    # # model.set_dist_func(gumbel_softmax_u)
    # rmse, mae = get_regression_performance(train, test, model, preprocessor, group_parameter, target)
    # print('RMSE {0}, MAE {1} (gumble, soft)'.format(rmse, mae))
