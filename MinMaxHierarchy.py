from ast import literal_eval as make_tuple
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from ast import literal_eval as make_tuple
from UltrametricMatrix import flatten


def MinMaxHierarchy(points, metric='euclidean'):
    print("VERSION 18")
    dist = pairwise_distances(points, metric=metric)
    for i in range(len(dist)):
        dist[i][i] = 0

    init_dist = dist.copy()
    
    
    print('Distance matrix: 0 step')
    print(pd.DataFrame(dist))
    
    dist[dist == 0] = np.max(dist) + 1
    
    
    clusters = [str(i) for i in range(len(dist))]
    init_clusters = tuple(map(int, clusters.copy())) # DELETE
    
    
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
        

        print(f'Distance matrix: {k + 1} step')
        print(pd.DataFrame(new_dist, columns=clusters, index=clusters))
        print("\n\n")

        new_dist[new_dist == 0] = np.max(new_dist) + 1

        dist = new_dist
    
    return make_tuple(clusters[0]), init_dist, ultra_dists