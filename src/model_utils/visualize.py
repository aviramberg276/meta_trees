import numpy as np
from graphviz import Digraph

NODE_ATTR = {'shape': 'box', 'style': 'rounded', 'fontsize': '18'}
LEAF_ATTR = {'shape': 'oval', 'width': '0.7', 'fixedsize': 'true', 'fontsize': '16'}
LEFT_STRING = '0'
RIGHT_STRING = '1'


def _get_top_weights(weights, threshold=0.9):
    weights_sum = sum([abs(w) for w in weights])
    sorted_weights = sorted([(w, i) for i, w in enumerate(weights)], key=lambda v: abs(v[0]))
    top_weights = []
    current_sum = 0
    while current_sum < weights_sum * threshold:
        weight = sorted_weights.pop()
        if weight[0] <= 0.02:
            break
        top_weights.append(weight)
        current_sum += abs(weight[0])
    return top_weights


def _get_node_rep(node, index, features, preprocessor, is_dummy):
    if node.is_leaf():
        return _get_leaf_rep(node, index)

    bias = node.biases.detach().cpu().numpy()[index]
    weights = node.weights.detach().cpu().numpy()[index]
    beta = node.beta.detach().cpu().numpy()
    if len(beta) > index:
        beta = beta[index]

    normed_bias = (np.ones_like(features).astype(float) * bias)
    normed_bias = preprocessor.inverse_transform(-normed_bias)
    bias = (weights * normed_bias).sum()

    top_weights = _get_top_weights(weights)
    feature_index = top_weights[0][1]
    feature = features[feature_index]
    if is_dummy[feature_index]:
        ratio = 'is' if beta > 0 else 'not'
        return '{0} {1}?'.format(ratio, feature)
    ratio = '>' if beta > 0 else '<'
    return '{0}{1}{2:.2f}'.format(feature, ratio, bias)


def _get_leaf_rep(node, index):
    preds = node.preds[index].detach().cpu().numpy()
    return ', '.join(['{0:.3f}'.format(p) for p in preds])


def render_tree(model, x_train, y_train, preprocessor, indices, features=None, is_dummy=None):
    if features is None:
        features = ['X_{0}'.format(str(i)) for i in range(x_train.shape[-1])]

    if is_dummy is None:
        is_dummy = [False for _ in range(x_train.shape[-1])]

    tree = model.get_tree(x_train, y_train)

    gs = Digraph()
    for index in indices:
        with gs.subgraph(name=str(index)) as g:
            root_name = '{0}_root'.format(index)

            nodes = []
            node_names = []
            nodes.append(tree)
            node_names.append(root_name)

            while nodes:
                node, node_name = nodes.pop(), node_names.pop()
                decision_func = _get_node_rep(node, index, features, preprocessor, is_dummy)
                g.node(node_name, decision_func, **NODE_ATTR)
                if node.is_leaf():
                    continue

                left_name = '{0}_left'.format(node_name)
                g.edge(node_name, left_name, LEFT_STRING)
                left_node = node.left
                if left_node.is_leaf():
                    probs = _get_leaf_rep(left_node, index)
                    g.node(left_name, probs, **LEAF_ATTR)
                else:
                    nodes.append(left_node)
                    node_names.append(left_name)

                right_name = '{0}_right'.format(node_name)
                g.edge(node_name, right_name, RIGHT_STRING)
                right_node = node.right
                if right_node.is_leaf():
                    probs = _get_leaf_rep(right_node, index)
                    g.node(right_name, probs, **LEAF_ATTR)
                else:
                    nodes.append(right_node)
                    node_names.append(right_name)
    return gs

