from MinMaxHierarchy import MinMaxHierarchy
from MedianHierarchy import MedianHierarchy
from Linkages import single_linked
from Linkages import complete_linked
from Linkages import group_average_linked
from Linkages import weighted_average_linked
from UltrametricMatrix import ultramatrix
from HierarchyClusteringWithoutLogs import hierarchy
import numpy as np
import scipy.stats as sts
import pandas as pd


FUNCOFMETHODS = {
    'single_linked': single_linked,
    'complete_linked': complete_linked,
    'group_average_linked': group_average_linked,
    'weighted_average_linked': weighted_average_linked,
    'min_max_linked': MinMaxHierarchy,
    'median_linked': MedianHierarchy
}


# Функция `pipe(points, method)` считает метрики исходя из заданных кластеров и методов
def pipe(points, method):
    if method.__name__ == 'MinMaxHierarchy':
        logs = MinMaxHierarchy(points, metric='euclidean')
    elif method.__name__ == 'MedianHierarchy':
        logs = MedianHierarchy(points, metric='euclidean')
    else:
        logs = hierarchy(points, metric='euclidean', method=method)
    
    start_matrix = logs[1]
    ultra_dists = logs[2]
    finish_matrix = ultramatrix(logs[0], ultra_dists)

    max_abs = np.max(np.abs(start_matrix - finish_matrix))

    n_points = len(start_matrix) 
    N_edge = n_points * (n_points - 1) / 2
    norm_sum_abs = np.sum(np.abs(finish_matrix - start_matrix)) / N_edge

    return max_abs, norm_sum_abs, ultra_dists


# Функция `get_gen_sample(size)` генерирует кластеры
def get_gen_sample(size):
    N = int(size / 3)

    norm1 = sts.norm(1, 0.2)
    norm2 = sts.norm(1.5, 0.1)
    norm3 = sts.norm(2, 0.2)

    x = np.append(norm1.rvs(N), np.append(norm2.rvs(N), norm3.rvs(N), axis=0), axis=0)
    y = np.append(norm1.rvs(N), np.append(norm2.rvs(N), norm3.rvs(N), axis=0), axis=0)

    points = list(zip(x, y))

    return np.array(points)


# Функция `times_when_method_better(results, res_column)` создает матрицу, которая показывает соотношение
# количества раз, когда метрика по указанному в строке методу оказалась меньше, чем метрика по указанному
# в столбце методу, к общему количеству экспериментов
def times_when_method_better(results):

    ResultsMatrix = pd.DataFrame(columns=results.columns, index=results.columns)

    for col in results.columns:
        for ind in results.columns:
            res = results[results[ind] < results[col]].shape[0] / results.shape[0]
            ResultsMatrix[col][ind] = res

    return ResultsMatrix.astype(float)


# Основная функция, которая запускает программу
def RunExperiment(size, sample_size, n_iter, FUNCOFMETHODS):

    MetricsByMethodsForMax = {method_name: [] for method_name in FUNCOFMETHODS.keys()}
    MetricsByMethodsForSum = {method_name: [] for method_name in FUNCOFMETHODS.keys()}

    if n_iter * sample_size >= size:
        return "n_iter * sample_size >= size, сделайтее size больше"
    
    points = get_gen_sample(size)
    ssamples = []
    all_ultradists = {name: [] for name in FUNCOFMETHODS.keys()}

    for _ in range(n_iter):
        indices = np.random.choice(points.shape[0], size=sample_size, replace=False)
        sample = points[indices]
        ssamples.append(sample)

        for method_name, method_func in FUNCOFMETHODS.items():
            metrics_both_and_ultradists = pipe(sample, method_func)
            MetricsByMethodsForMax[method_name].append(metrics_both_and_ultradists[0])
            MetricsByMethodsForSum[method_name].append(metrics_both_and_ultradists[1])
            all_ultradists[method_name].append(metrics_both_and_ultradists[2])

    ResultsForMax = pd.DataFrame(MetricsByMethodsForMax)
    ResultsForSum = pd.DataFrame(MetricsByMethodsForSum)

    return ResultsForMax, ResultsForSum, ssamples, all_ultradists

