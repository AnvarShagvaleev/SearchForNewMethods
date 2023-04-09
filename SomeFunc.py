# Просто функция для включения и выключения логов
def nprint(text, switch):
    if switch == True:
        print(text)


def comparison_of_two(ResultsForMetric, Samples, mone, mtwo, metric='max'):
    if metric == 'max':
        print("Метрика 'Максимального отклонения'")
    elif metric == "sum":
        print("Метрика 'Взвешенная сумма отклонений'")
    else:
        return "Нет такой метрики"
    print(f"Сэмплы, когда {mone} дает результаты хуже, чем {mtwo}")
    inds = list(ResultsForMetric[ResultsForMetric[mone] >= ResultsForMetric[mtwo]].index)
    flag = 0
    for k, sample in enumerate(Samples):
        if k in inds:
            print(f"Номер выборки {k}")
            print(sample)
            print()
            flag = 1
    if flag != 1:
        print("Случаев не обнаружено")