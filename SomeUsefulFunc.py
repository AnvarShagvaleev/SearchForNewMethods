from Hierarchy.PipelineFunctions.MainWorkers import times_when_method_better, MakeDendogram, generator, RunExperiment, ReadLogs
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
import pickle


def parseParams(direction_name):
    file_name = './LOGS/' + direction_name + '/Samples'
    main_info_about_sample = direction_name.split()[0].split("-")

    dim = int(main_info_about_sample[0][0])
    sample_size = int(main_info_about_sample[1])
    iter_number_def = main_info_about_sample[2]

    if 'k' in iter_number_def:
        iter_number = int(float(iter_number_def[:-1])) * 1000
    else:
        iter_number = int(float(iter_number_def))

    return dim, sample_size, iter_number, direction_name, file_name

def getSamples(file_name):
    file_samples = open(file_name, 'rb')
    Samples = pickle.load(file_samples)

    return Samples

def getResultsList(direction_name):
    ResultsForMaxList = {}
    ResultsForSumList = {}
    for dist_metric in ('euclidean', 'cityblock', 'chebyshev'):
        dist_metric_dir = f"./LOGS/{direction_name}/{dist_metric}"
        
        TimeLogsList, Ultradists, MaxList, SumList, NameOfMethodList = ReadLogs(dist_metric_dir)

        MaxMatrix = pd.DataFrame({'MetricsByMethodsForMax': MaxList, 'NameOfMethod': NameOfMethodList})
        grouper = MaxMatrix.groupby('NameOfMethod')
        ResultsForMax = pd.concat([pd.Series(v['MetricsByMethodsForMax'].tolist(), name=k) for k, v in grouper], axis=1)
        ResultsForMaxList[dist_metric] = ResultsForMax

        SumMatrix = pd.DataFrame({'MetricsByMethodsForSum': SumList, 'NameOfMethod': NameOfMethodList})
        grouper = SumMatrix.groupby('NameOfMethod')
        ResultsForSum = pd.concat([pd.Series(v['MetricsByMethodsForSum'].tolist(), name=k) for k, v in grouper], axis=1)
        ResultsForSumList[dist_metric] = ResultsForSum
    
    return ResultsForMaxList, ResultsForSumList

def getPlot(ResultList, dim, sample_size, iter_number, exp_metric_name):
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(20,5))

    for k, (dist_metric, res) in enumerate(ResultList.items()):
        if k == 0:
            sns.heatmap(data=times_when_method_better(res), ax=axes[k], annot=True, cbar=False)
            axes[k].set_title(dist_metric)
        else:
            sns.heatmap(data=times_when_method_better(res), ax=axes[k], annot=True, cbar=False, yticklabels=False)
            axes[k].set_title(dist_metric)

    metric_caption = 'Максимальное отклонение' if exp_metric_name == 'max' else 'Сумма отклонений'
    title = f'{metric_caption}, ({dim} dim: {sample_size} точек - {iter_number} опытов)'

    fig.suptitle(title, fontsize=16)
    plt.savefig(f'./Plots/{title}')
    plt.show()
