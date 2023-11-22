# %%
import numpy as np
from itertools import combinations
from scipy.stats import mannwhitneyu
import json
from pathlib import Path
data_list = []
path_graph_dir_path = Path(r'C:\Users\OV\Desktop\Uni\Master_Thesis_Project\experiment_results\Mu Plus One GA Complete Bipartite 2')
for file_path in path_graph_dir_path.glob('*.json'):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        data_list.append(data)
inv_2 = {}
inv_5 = {}
inv_10 = {}
uni_2 = {}
uni_5 = {}
uni_10 = {}
uni_2_true = {}
uni_5_true = {}
uni_10_true = {}
for exp in data_list:
    graph_size = exp['Parameters'][1]
    crossover_operator = exp['Parameters'][2]
    population = exp['Parameters'][5]
    optimization_steps = exp['OptimizationSteps']
    div_mechanism = exp['Parameters'][6]
    if crossover_operator == "Uniform":
        if div_mechanism == True:
            if population == 2:
                uni_2_true[graph_size] = optimization_steps
            elif population == 5:
                uni_5_true[graph_size] = optimization_steps
            elif population == 10:
                uni_10_true[graph_size] = optimization_steps
        elif div_mechanism == False:
            if population == 2:
                uni_2[graph_size] = optimization_steps
            elif population == 5:
                uni_5[graph_size] = optimization_steps
            elif population == 10:
                uni_10[graph_size] = optimization_steps
    elif crossover_operator == "Inverse":
        if population == 2:
            inv_2[graph_size] = optimization_steps
        elif population == 5:
            inv_5[graph_size] = optimization_steps
        elif population == 10:
            inv_10[graph_size] = optimization_steps

crossovers=['Uniform', 'Div-Uniform', 'Inverse']
results_2 = {}
hypotheses_false = []
hypotheses_true = []
for graph_size in sorted(inv_2.keys()):
    stat, p_value_false = mannwhitneyu(inv_2[graph_size], uni_2[graph_size], alternative='less')
    stat, p_value_true = mannwhitneyu(inv_2[graph_size], uni_2_true[graph_size], alternative='less')
    hypotheses_false.append(f"{p_value_false:.2e}")
    hypotheses_true.append(f"{p_value_true:.2e}")
results_2[0] = hypotheses_false
results_2[1] = hypotheses_true


results_5 = {}
hypotheses_false = []
hypotheses_true = []
for graph_size in sorted(inv_2.keys()):
    stat, p_value_false = mannwhitneyu(inv_5[graph_size], uni_5[graph_size], alternative='less')
    stat, p_value_true = mannwhitneyu(inv_5[graph_size], uni_5_true[graph_size], alternative='less')
    hypotheses_false.append(f"{p_value_false:.2e}")
    hypotheses_true.append(f"{p_value_true:.2e}")
results_5[0] = hypotheses_false
results_5[1] = hypotheses_true

results_10 = {}
hypotheses_false = []
hypotheses_true = []
for graph_size in sorted(inv_2.keys()):
    stat, p_value_false = mannwhitneyu(inv_10[graph_size], uni_10[graph_size], alternative='less')
    stat, p_value_true = mannwhitneyu(inv_10[graph_size], uni_10_true[graph_size], alternative='less')
    hypotheses_false.append(f"{p_value_false:.2e}")
    hypotheses_true.append(f"{p_value_true:.2e}")
results_10[0] = hypotheses_false
results_10[1] = hypotheses_true
# %%
