from Hierarchy.MinMaxMethod import MinMaxHierarchy
from Hierarchy.MedianMethod import MedianHierarchy
from Hierarchy.ToCulcMethods.Linkages import single_linked, complete_linked, group_average_linked, weighted_average_linked
from Hierarchy.ToCulcMethods.UltrametricMatrix import ultramatrix
from Hierarchy.StandartMethods import hierarchy
import numpy as np
import scipy.stats as sts
import pandas as pd
import pickle
import os
import datetime
from tqdm import tqdm
import time
from scipy.cluster import hierarchy as scipy_hierarchy
import matplotlib.pyplot as plt


FUNCOFMETHODS = {
    'single_linked': single_linked,
    'complete_linked': complete_linked,
    'group_average_linked': group_average_linked,
    'weighted_average_linked': weighted_average_linked,
    'min_max_linked': MinMaxHierarchy,
    'median_linked': MedianHierarchy
}


# Функция `pipe(points, method)` считает метрики исходя из заданных кластеров и методов
def pipe(points, method, dist_metric):
    if method.__name__ == 'MinMaxHierarchy':
        logs = MinMaxHierarchy(points, metric=dist_metric)
    elif method.__name__ == 'MedianHierarchy':
        logs = MedianHierarchy(points, metric=dist_metric)
    else:
        logs = hierarchy(points, metric=dist_metric, method=method)
    
    start_matrix = logs[1]
    ultra_dists = logs[2]
    finish_matrix = ultramatrix(logs[0], ultra_dists)

    diff = np.abs(start_matrix - finish_matrix)

    max_abs = np.max(diff)

    n_points = len(start_matrix) 
    N_edge = n_points * (n_points - 1) / 2
    norm_sum_abs = np.sum(diff) / N_edge

    start_matrix_copy = start_matrix.copy()
    start_matrix_copy[start_matrix_copy == 0] = 1
    norm_diff = diff / start_matrix_copy

    return max_abs, norm_sum_abs, ultra_dists, norm_diff


# Функция `get_gen_sample(size, func)` генерирует кластеры
def get_gen_sample(size, func, dim):

    res = func(size, dim)

    if type(res) == np.ndarray:
        return res
    else:
        return "Type of function func must be numpy.ndarray"


# Функция `times_when_method_better(results, res_column)` создает матрицу, которая показывает соотношение
# количества раз, когда метрика по указанному в строке методу оказалась меньше, чем метрика по указанному
# в столбце методу, к общему количеству экспериментов
def times_when_method_better(results):

    ResultsMatrix = pd.DataFrame(columns=results.columns, index=results.columns)

    for col in results.columns:
        for ind in results.columns:
            res = results[results[ind] <= results[col]].shape[0] / results.shape[0]
            ResultsMatrix[col][ind] = res

    return ResultsMatrix.astype(float)


def MakeDendogram(sample, ultras):
    temp = scipy_hierarchy.linkage(sample, 'single')
    temp[:, 2] = ultras
    plt.figure()
    dn = scipy_hierarchy.dendrogram(temp)

    for i, d in zip(dn['icoord'], dn['dcoord']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        plt.plot(x, y, 'ro')
        plt.annotate(y, (x, y), xytext=(0, -8),
                        textcoords='offset points',
                        va='top', ha='center')


# Основная функция, которая запускает программу
def NewRunExperiment(size, func, sample_size, n_iter, FUNCOFMETHODS, dist_metric, dim):

    if n_iter * sample_size >= size:
        return "n_iter * sample_size >= size, сделайте size больше"
    
    points = get_gen_sample(size, func, dim)


    n_iter_format = str(n_iter) if n_iter < 1000 else f"{n_iter / 1000}k"
    now = f"{str(dim)+'dim'}-{dist_metric}-{sample_size}-{n_iter_format} {str(datetime.datetime.today().replace(microsecond=0))}"
    os.mkdir(f"./LOGS/{now}")

    F_Samples = open(f"./LOGS/{now}/Samples", 'wb')
    F_MetricsByMethodsForMax = open(f"./LOGS/{now}/MetricsByMethodsForMax", 'wb')
    F_MetricsByMethodsForSum = open(f"./LOGS/{now}/MetricsByMethodsForSum", 'wb')
    F_Ultradists = open(f"./LOGS/{now}/Ultradists", 'wb')
    F_NameOfMethod = open(f"./LOGS/{now}/NameOfMethod", 'wb')
    F_TimeLogs = open(f"./LOGS/{now}/TimeLogs", 'wb')
    F_NormDiff = open(f"./LOGS/{now}/NormDiff", "wb")

    for _ in tqdm(range(n_iter)):
        indices = np.random.choice(points.shape[0], size=sample_size, replace=False)
        sample = points[indices]

        for method_name, method_func in FUNCOFMETHODS.items():
            tp1 = time.time()
            metrics_both_and_ultradists = pipe(sample, method_func, dist_metric)
            tp2 = time.time()

            pickle.dump(sample, F_Samples) # - тут переместить
            pickle.dump(metrics_both_and_ultradists[0], F_MetricsByMethodsForMax)
            pickle.dump(metrics_both_and_ultradists[1], F_MetricsByMethodsForSum)
            pickle.dump(metrics_both_and_ultradists[2], F_Ultradists)
            pickle.dump(metrics_both_and_ultradists[3], F_NormDiff)
            pickle.dump(method_name, F_NameOfMethod)
            pickle.dump(tp2-tp1, F_TimeLogs)

    return now


def RunExperiment(size, func, sample_size, n_iter, FUNCOFMETHODS, dist_metric):

    MetricsByMethodsForMax = {method_name: [] for method_name in FUNCOFMETHODS.keys()}
    MetricsByMethodsForSum = {method_name: [] for method_name in FUNCOFMETHODS.keys()}

    if n_iter * sample_size >= size:
        return "n_iter * sample_size >= size, сделайтее size больше"
    
    points = get_gen_sample(size, func)
    Samples = []
    all_ultradists = {name: [] for name in FUNCOFMETHODS.keys()}

    for _ in range(n_iter):
        indices = np.random.choice(points.shape[0], size=sample_size, replace=False)
        sample = points[indices]
        Samples.append(sample)

        for method_name, method_func in FUNCOFMETHODS.items():
            metrics_both_and_ultradists = pipe(sample, method_func, dist_metric)
            MetricsByMethodsForMax[method_name].append(metrics_both_and_ultradists[0])
            MetricsByMethodsForSum[method_name].append(metrics_both_and_ultradists[1])
            all_ultradists[method_name].append(metrics_both_and_ultradists[2])

    ResultsForMax = pd.DataFrame(MetricsByMethodsForMax)
    ResultsForSum = pd.DataFrame(MetricsByMethodsForSum)

    return ResultsForMax, ResultsForSum, Samples, all_ultradists



def RunCheck(Samples, FUNCOFMETHODS, dist_metric):

    MetricsByMethodsForMax = {method_name: [] for method_name in FUNCOFMETHODS.keys()}
    MetricsByMethodsForSum = {method_name: [] for method_name in FUNCOFMETHODS.keys()}
    all_ultradists = {name: [] for name in FUNCOFMETHODS.keys()}

    for sample in Samples:
        for method_name, method_func in FUNCOFMETHODS.items():
            metrics_both_and_ultradists = pipe(sample, method_func, dist_metric)
            MetricsByMethodsForMax[method_name].append(metrics_both_and_ultradists[0])
            MetricsByMethodsForSum[method_name].append(metrics_both_and_ultradists[1])
            all_ultradists[method_name].append(metrics_both_and_ultradists[2])

    ResultsForMax = pd.DataFrame(MetricsByMethodsForMax)
    ResultsForSum = pd.DataFrame(MetricsByMethodsForSum)

    return ResultsForMax, ResultsForSum, Samples, all_ultradists


def ReadLogs(now):
    file_names = (
        'TimeLogs',
        'Ultradists',
        'Samples',
        'MetricsByMethodsForMax',
        'MetricsByMethodsForSum',
        'NameOfMethod',
        'NormDiff'
    )

    filedata = dict()

    for name in file_names:
        data_from_file = open(f'./LOGS/{now}/{name}', 'rb')
        data_list = []
        while 1:
            try:
                data_list.append(pickle.load(data_from_file))
            except EOFError:
                break
        
        filedata[name] = data_list
    
    return filedata['TimeLogs'], filedata['Ultradists'], filedata['Samples'], filedata['MetricsByMethodsForMax'], filedata['MetricsByMethodsForSum'], filedata['NameOfMethod'], filedata['NormDiff']