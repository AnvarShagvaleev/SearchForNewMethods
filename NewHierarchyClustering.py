from Linkages import single_linked
from ast import literal_eval as make_tuple
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from UltrametricMatrix import count_clusters
from SomeFunc import nprint
from UltrametricMatrix import flatten


def hierarchy(points, metric='euclidean', method=single_linked, logs_turn_on=False):
    dist = pairwise_distances(points, metric=metric)
    for i in range(len(dist)):
        dist[i][i] = 0

    init_dist = dist.copy()


    nprint('Distance matrix: 0 step', logs_turn_on)
    nprint(pd.DataFrame(dist), logs_turn_on)

    dist[dist == 0] = np.max(dist) + 1
    
    
    clusters = [str(i) for i in range(len(dist))]
    
    
    dtype = '<U' + str(5 * sum([len(i) for i in clusters]))
    clusters = np.array(clusters, dtype=dtype)

    nprint(f"\nClusters:{clusters}\n\n", logs_turn_on)

    ultra_dists = []
    for k in range(len(dist) - 1):
        indices = np.unravel_index(np.argmin(dist), dist.shape)

        ultra_dists.append(dist[indices])
        
        
        c1 = clusters[indices[0]]
        c2 = clusters[indices[1]]
        new_cluster = f'({c1}, {c2})'
        clusters = np.delete(clusters, indices)
        clusters = np.insert(clusters, 0, new_cluster, axis=0)
        nprint(('Clusters:', clusters), logs_turn_on)

        
        new_dist = np.delete(dist, indices[0], axis=0)
        new_dist = np.delete(new_dist, indices[1] - 1, axis=0)
        new_dist = np.delete(new_dist, indices[0], axis=1)
        new_dist = np.delete(new_dist, indices[1] - 1, axis=1)
        new_dist = np.insert(new_dist, 0, 0, axis=1)
        new_dist = np.insert(new_dist, 0, 0, axis=0)


        if method.__name__ in ('MinMaxHierarchy', 'MedianHierarchy'):
            ###NEW
            new_dist = pd.DataFrame(new_dist, columns=clusters, index=clusters)

            for p, other_cluster in enumerate(new_dist.columns):
                options = []
                if p != 0:
                    if type(other_cluster) == str:
                        other_cluster = flatten(make_tuple(other_cluster))
                        for j in other_cluster:
                            for i in flatten(make_tuple(new_dist.columns[0])):
                                options.append(init_dist[i][j])
                    else:
                        for i in make_tuple(new_dist.columns[0]):
                            options.append(init_dist[i][other_cluster])
                    new_dist.iloc[0, p] = new_dist.iloc[p, 0] = (np.min(options) + np.max(options)) / 2

            new_dist = np.array(new_dist)
            

            for i in range(len(new_dist)):
                new_dist[i][i] = 0
            

            nprint(f'Distance matrix: {k + 1} step', logs_turn_on)
            nprint(pd.DataFrame(new_dist, columns=clusters, index=clusters), logs_turn_on)
            nprint("\n\n", logs_turn_on)

            new_dist[new_dist == 0] = np.max(new_dist) + 1

            dist = new_dist
        else:
            ###OLD
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

            dist_for_log = new_dist.copy()
            for i in range(len(dist_for_log)):
                dist_for_log[i][i] = 0
            nprint(f'Distance matrix: {k + 1} step', logs_turn_on)
            nprint(pd.DataFrame(dist_for_log), logs_turn_on)
            nprint("\n\n", logs_turn_on)

            dist = new_dist

        ###
    
    return make_tuple(clusters[0]), init_dist, ultra_dists