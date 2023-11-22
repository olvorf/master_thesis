# %%
import numpy as np
from itertools import combinations
from scipy.stats import mannwhitneyu
import json
from pathlib import Path
data_list = []
path_graph_dir_path = Path(r'C:\Users\OV\Desktop\Uni\Master_Thesis_Project\experiment_results\Complete Bipartite 2')
for file_path in path_graph_dir_path.glob('*.json'):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        data_list.append(data)
standard_mutation = {}
heavy_tailed_mutation = {}
biased_mutation = {}
flip_random_path_mutation = {}
biased_flip_random_path_mutation = {}
for exp in data_list:
    # graph_class = exp['Parameters'][1]
    graph_size = exp['Parameters'][1]
    mutation_operator = exp['Parameters'][2]
    beta = exp['Parameters'][3]
    avg_optimization_steps = exp['AverageOptimizationSteps']
    optimization_steps = exp['OptimizationSteps']
    if mutation_operator == "Standard Mutation":
        standard_mutation[graph_size] = optimization_steps
    elif mutation_operator == "Heavy Tailed Mutation":
        heavy_tailed_mutation[graph_size] = optimization_steps
    elif mutation_operator == "Biased Mutation":
        biased_mutation[graph_size] = optimization_steps
    elif mutation_operator == "Flip Random Path Mutation":
        flip_random_path_mutation[graph_size] = optimization_steps
    elif mutation_operator == "Biased Flip Random Path Mutation":
        biased_flip_random_path_mutation[graph_size] = optimization_steps
data_new = {}

data_new['Heavy Tailed Mutation'] = heavy_tailed_mutation
data_new['Standard Mutation'] = standard_mutation
data_new['Flip Random Path Mutation'] = flip_random_path_mutation
data_new['Biased Flip Random Path Mutation'] = biased_flip_random_path_mutation
data_new['Biased Mutation'] = biased_mutation

algorithms = list(data_new.keys())
num_algorithms = len(algorithms)
p_values = np.zeros((num_algorithms, num_algorithms))
alpha = 0.05
results = {}
for i, j in combinations(range(num_algorithms), 2):
    hypotheses = []
    for size in sorted(data_new[algorithms[0]].keys()):
        stat, p_value = mannwhitneyu(data_new[algorithms[j]][size], data_new[algorithms[i]][size], alternative='less')
        hypotheses.append(f"{p_value:.2e}")
    results[(j, i)] = hypotheses
print(results)
# %%
