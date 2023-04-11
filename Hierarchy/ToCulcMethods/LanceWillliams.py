def LanceWillliams(dist, i, j, k, alpha_u, alpha_v, beta, gamma):
    return alpha_u * dist[i][k] + alpha_v * dist[j][k] + \
    beta * dist[i][j] + gamma * abs(dist[i][k] - dist[j][k])