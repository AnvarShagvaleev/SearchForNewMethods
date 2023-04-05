from LanceWillliams import LanceWillliams

def single_linked(dist, i, j, k, clusters_param=0):
    alpha_u, alpha_v, beta, gamma = 0.5, 0.5, 0, -0.5
    return LanceWillliams(dist, i, j, k, alpha_u, alpha_v, beta, gamma)

def complete_linked(dist, i, j, k, clusters_param=0):
    alpha_u, alpha_v, beta, gamma = 0.5, 0.5, 0, 0.5
    return LanceWillliams(dist, i, j, k, alpha_u, alpha_v, beta, gamma)

def group_average_linked(dist, i, j, k, clusters_param):
    alpha_u, alpha_v, beta, gamma = clusters_param[0] / clusters_param[2], clusters_param[1] / clusters_param[2], 0, 0
    return LanceWillliams(dist, i, j, k, alpha_u, alpha_v, beta, gamma)

def weighted_average_linked(dist, i, j, k, clusters_param=0):
    alpha_u, alpha_v, beta, gamma = 0.5, 0.5, 0, 0
    return LanceWillliams(dist, i, j, k, alpha_u, alpha_v, beta, gamma)

# def ward(dist, i, j, k, clusters_param):
#     alpha_u, alpha_v, beta, gamma = clusters_param[0] / clusters_param[2], clusters_param[1] / clusters_param[2], 0, 0
#     return LanceWillliams(dist, i, j, k, alpha_u, alpha_v, beta, gamma)

# def min_max_linked(dist, i, j, k, clusters_param):
#     return (single_linked(dist, i, j, k, clusters_param) + complete_linked(dist, i, j, k, clusters_param)) / 2