import random
import time
import copy
import matplotlib.pyplot as plt
import numpy as np


def selection_sort(array):
    comparison = 0
    for i in range(len(array) - 1):
        minimum = i
        for j in range(i + 1, len(array)):
            comparison += 1
            if array[j] < array[minimum]: minimum = j
        to_change = array[i]
        array[i] = array[minimum]
        array[minimum] = to_change
    return array, comparison


def insertion_sort(array):
    comparison = 0
    for i in range(1, len(array)):
        start = array[i]
        comparison += 1
        while i > 0 and array[i] < array[i - 1]:
            comparison += 1
            array[i] = array[i - 1]
            array[i - 1] = start
            i -= 1
    return array, comparison


def shell_sort(array):
    gap = len(array) // 2
    comparison = 0
    while gap > 0:
        for i in range(len(array) - gap):
            comparison += 1
            start = array[i + gap]
            start_index = i + gap
            while start_index >= gap and array[start_index] < array[start_index - gap]:
                comparison += 1
                array[start_index] = array[start_index - gap]
                array[start_index - gap] = start
                start_index -= gap

        gap = gap // 2
    return array, comparison


def first_experiment(x):
    random_array = [random.randint(0, 1000) for _ in range(pow(2, 7 + x))]
    return random_array


def second_experiment(x):
    random_array = [random.randint(0, 1000) for _ in range(pow(2, 7 + x))]
    random_array.sort()
    return random_array


def third_experiment(x):
    random_array = [random.randint(0, 1000) for _ in range(pow(2, 7 + x))]
    random_array.sort(reverse=True)
    return random_array


def fourth_experiment(x):
    random_array = [random.randint(1, 3) for _ in range(pow(2, 7 + x))]
    return random_array


def main():
    algorithms = {selection_sort: 'Selection Sort', insertion_sort: 'Insertion Sort', shell_sort: 'Shell Sort'}
    experiments = [first_experiment, second_experiment, third_experiment, fourth_experiment]
    for experiment in experiments:
        with open('algorithms.txt', 'a+') as file:
            print(f'EXPERIMENT {experiments.index(experiment) + 1}\n', file=file)
        print(f'EXPERIMENT {experiments.index(experiment) + 1}\n')
        for x in range(14):
            with open('algorithms.txt', 'a+') as file:
                generated_array = experiment(x)
                if experiment == first_experiment or experiment == fourth_experiment:
                    generated_arrays = [experiment(x) for _ in range(10)]
                    for algorithm in algorithms:
                        print(f'Working on {algorithms[algorithm]} with the power {7 + x} and experiment '
                              f'{experiments.index(experiment) + 1}\n')
                        print('\n' + algorithms[algorithm] + '\n', file=file)
                        average_timing = 0
                        average_comparison = 0
                        for i in range(10):
                            print(f'ITERATION {i + 1}\n-----------------------------------------------', file=file)
                            start = time.time()
                            algo = algorithm(copy.deepcopy(generated_arrays)[i])
                            average_timing += time.time() - start
                            average_comparison += algo[1]
                        print(f'\n\nTime spent on {algorithms[algorithm]} (average) with arrays to the power of {7 + x}: '
                              f'{round(average_timing / 10, 5)}', file=file)
                        print(f'\n{algorithms[algorithm]} (average) took {average_comparison / 10} number of '
                              f'comparison operations\n', file=file)

                else:
                    for algorithm in algorithms:
                        print(f'Working on {algorithms[algorithm]} with the power {7 + x} and experiment '
                              f'{experiments.index(experiment) + 1}\n')
                        print('\n' + algorithms[algorithm] + '\n', file=file)
                        start = time.time()
                        algo = algorithm(copy.deepcopy(generated_array))
                        print(f'\nTime spent on {algorithms[algorithm]} with arrays to the power of {7 + x}: '
                              f'{round(time.time() - start, 5)}\n', file=file)
                        print(f'\n{algorithms[algorithm]} took {algo[1]} number of '
                              f'comparison operations\n', file=file)


def data_preparation():
    with open('algorithms.txt') as file:
        lines = file.readlines()
        algo_dict = {'Timing': [[line.replace('(average) ', '').split()[-10], line.replace('(average) ', '').split()[-9],
                                 line.replace('(average) ', '').split()[-2], line.replace('(average) ', '').split()[-1]]
                                for line in lines if line.startswith('Time')],
                     'Comparison': [[line.replace('(average) ', '').split()[-8], line.replace('(average) ', '').split()[-7],
                                 line.replace('(average) ', '').split()[-5]]
                                 for line in lines if line.endswith('operations\n')]}
        # for i in algo_dict['Comparison']:
        #
        #     print(i)
        data = {'Insertion': {'Experiment 1': [], 'Experiment 2': [], 'Experiment 3': [], 'Experiment 4': []},
                'Selection': {'Experiment 1': [], 'Experiment 2': [], 'Experiment 3': [], 'Experiment 4': []},
                'Shell': {'Experiment 1': [], 'Experiment 2': [], 'Experiment 3': [], 'Experiment 4': []}}

        sorted = list(zip(algo_dict['Timing'], algo_dict['Comparison']))
        sorted.sort(key=lambda x: x[0][0])
        all = ['Insertion', 'Selection', 'Shell']
        i = 0
        counter = 0
        experiment = 1
        for each in sorted:
            if counter == 11: counter = 0; experiment += 1
            if experiment == 5: experiment = 1; i += 1

            data[all[i]][f'Experiment {experiment}'].append(each)
            counter += 1
        return data


def analysis(data):
    experiment_1 = {'Insertion': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Insertion']['Experiment 1']],
                                  'Операції порівняння': [float(y[1][-1]) for y in data['Insertion']['Experiment 1']]},
                    'Selection': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Selection']['Experiment 1']],
                                  'Операції порівняння': [float(y[1][-1]) for y in data['Selection']['Experiment 1']]},
                    'Shell': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Shell']['Experiment 1']],
                              'Операції порівняння': [float(y[1][-1]) for y in data['Shell']['Experiment 1']]
                              }}

    experiment_2 = {'Insertion': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Insertion']['Experiment 2']],
                                      'Операції порівняння': [float(y[1][-1]) for y in data['Insertion']['Experiment 2']]},
                        'Selection': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Selection']['Experiment 2']],
                                      'Операції порівняння': [float(y[1][-1]) for y in data['Selection']['Experiment 2']]},
                        'Shell': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Shell']['Experiment 2']],
                                  'Операції порівняння': [float(y[1][-1]) for y in data['Shell']['Experiment 2']]
                                  }}

    experiment_3 = {'Insertion': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Insertion']['Experiment 3']],
                                      'Операції порівняння': [float(y[1][-1]) for y in data['Insertion']['Experiment 3']]},
                        'Selection': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Selection']['Experiment 3']],
                                      'Операції порівняння': [float(y[1][-1]) for y in data['Selection']['Experiment 3']]},
                        'Shell': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Shell']['Experiment 3']],
                                  'Операції порівняння': [float(y[1][-1]) for y in data['Shell']['Experiment 3']]
                                  }}

    experiment_4 = {'Insertion': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Insertion']['Experiment 4']],
                                      'Операції порівняння': [float(y[1][-1]) for y in data['Insertion']['Experiment 4']]},
                        'Selection': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Selection']['Experiment 4']],
                                      'Операції порівняння': [float(y[1][-1]) for y in data['Selection']['Experiment 4']]},
                        'Shell': {'Час':[float(y[0][-1]) * pow(10, 9) for y in data['Shell']['Experiment 4']],
                                  'Операції порівняння': [float(y[1][-1]) for y in data['Shell']['Experiment 4']]
                                  }}

    print(experiment_3['Insertion']['Операції порівняння'])
    print(experiment_3['Selection']['Операції порівняння'])
    experiments_x = [x for x in range(7, 18)]

    i = 1
    for experiment in [experiment_1, experiment_2, experiment_3, experiment_4]:
        for operation in ['Час', 'Операції порівняння']:

            plt.figure(figsize=(20, 10), dpi=70)

            plt.title(f'Експеримент №{i}')
            plt.plot(experiments_x, experiment['Insertion'][operation])
            plt.plot(experiments_x, experiment['Selection'][operation])
            plt.plot(experiments_x, experiment['Shell'][operation])
            plt.xlabel('Cтепінь')
            plt.ylabel(f'{operation}')
            plt.gca().legend(('Insertion Sort', 'Selection Sort', 'Shell Sort'))
            plt.yscale('log')
            # plt.xscale('log')
            # xinterval = np.arange(0, pow(10))
            plt.savefig(f'ex_{i}_{operation}.png')
        i += 1


if __name__ == '__main__':
    analysis(data_preparation())
    # data_preparation()