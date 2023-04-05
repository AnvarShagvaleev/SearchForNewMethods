from Linkages import single_linked
from ast import literal_eval as make_tuple
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from UltrametricMatrix import count_clusters


def hierarchy(points, metric='euclidean', method=single_linked):
    dist = pairwise_distances(points, metric=metric)
    for i in range(len(dist)):
        dist[i][i] = 0

    init_dist = dist.copy()
    
    print('Distance matrix: 0 step')
    print(pd.DataFrame(dist))
    
    dist[dist == 0] = np.max(dist) + 1
    
    
    clusters = [str(i) for i in range(len(dist))]
    
    
    dtype = '<U' + str(5 * sum([len(i) for i in clusters]))
    clusters = np.array(clusters, dtype=dtype)

    print(f"\nClusters:{clusters}\n\n")
    
    ultra_dists = []
    for k in range(len(dist) - 1):
        indices = np.unravel_index(np.argmin(dist), dist.shape)

        ultra_dists.append(dist[indices])
        
        
        c1 = clusters[indices[0]]
        c2 = clusters[indices[1]]
        new_cluster = f'({c1}, {c2})'
        print(new_cluster)
        clusters = np.delete(clusters, indices)
        clusters = np.insert(clusters, 0, new_cluster, axis=0)
        print('Clusters:', clusters)

        
        new_dist = np.delete(dist, indices[0], axis=0)
        new_dist = np.delete(new_dist, indices[1] - 1, axis=0)
        new_dist = np.delete(new_dist, indices[0], axis=1)
        new_dist = np.delete(new_dist, indices[1] - 1, axis=1)

        
        new_dist = np.insert(new_dist, 0, 0, axis=1)
        new_dist = np.insert(new_dist, 0, 0, axis=0)

        
        cur_index = 0
        for i in range(len(dist)):
            if i not in indices:
                cur_index += 1
                new_dist[0][cur_index] = new_dist[cur_index][0] = method(
                    dist,
                    indices[0],
                    indices[1], 
                    i,
                    clusters_param=count_clusters(make_tuple(new_cluster))
                )

        new_dist[0][0] = np.max(new_dist) + 1

        print(f'Distance matrix: {k + 1} step')
        print(pd.DataFrame(new_dist))
        print("\n\n")

        dist = new_dist
    
    return make_tuple(clusters[0]), init_dist, ultra_dists