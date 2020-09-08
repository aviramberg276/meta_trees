from data.data_getter import get_movie_lens_100k, get_movie_lens_1m, get_jester
from model_utils.utils import DIST_SOFT, DIST_HARD


def get_dist_func(dist_func_name):
    if dist_func_name == 'soft':
        return DIST_SOFT
    return DIST_HARD


def get_data(data_name, subset=True):
    if data_name == 'ml_100k':
        return get_movie_lens_100k(subset)
    elif data_name == 'ml_1m':
        return get_movie_lens_1m(subset)
    return get_jester(subset)
