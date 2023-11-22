#%%
import json
from pathlib import Path

# Specify the directory where your JSON files are located
directory_path = Path(r'C:\Users\OV\Desktop\Uni\Master_Thesis_Project\experiment_results\Complete Bipartite 2')
GRAPH = "Complete Bipartite"
FONT_SIZE = 17
AXIS_FONT_SIZE = 16
output_dir = r"C:\Users\OV\Desktop\Uni\Master_Thesis_Project\result_analysis\mutation_result_images"
# Create a list to store the data from the JSON files
data_list = []

# Use a loop to iterate through the files in the directory and read each JSON file
for file_path in directory_path.glob('*.json'):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        data_list.append(data)

# %%
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
    if mutation_operator == "Standard Mutation":
        standard_mutation[graph_size] = avg_optimization_steps
    elif mutation_operator == "Heavy Tailed Mutation":
        heavy_tailed_mutation[graph_size] = avg_optimization_steps
    elif mutation_operator == "Biased Mutation":
        biased_mutation[graph_size] = avg_optimization_steps
    elif mutation_operator == "Flip Random Path Mutation":
        flip_random_path_mutation[graph_size] = avg_optimization_steps
    elif mutation_operator == "Biased Flip Random Path Mutation":
        biased_flip_random_path_mutation[graph_size] = avg_optimization_steps
# %%

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
x = list(standard_mutation.keys())  # Assuming all dictionaries have the same keys

x = sorted(x)

# Create lists for each dataset and sort them based on x_sorted
standard_mutation_values = [standard_mutation[key] for key in x]
heavy_tailed_mutation_values = [heavy_tailed_mutation[key] for key in x]
biased_mutation_values = [biased_mutation[key] for key in x]
flip_random_path_mutation_values = [flip_random_path_mutation[key] for key in x]
biased_flip_random_path_mutation_values = [biased_flip_random_path_mutation[key] for key in x]

plt.figure(figsize=(8, 5),dpi=350)  # Optional: Set the figure size
plt.plot(x, standard_mutation_values, linestyle='-',label='standard mut.', marker='o')
plt.plot(x, heavy_tailed_mutation_values, linestyle='-',label='fmut', marker='s')
plt.plot(x, biased_mutation_values, linestyle='-',label='biased mut.', marker='^')
plt.plot(x, flip_random_path_mutation_values, linestyle='-',label='FRP mut.', marker='x')
plt.plot(x, biased_flip_random_path_mutation_values, linestyle='-', label='biased FRP mut.', marker='D')
plt.grid(True)
plt.xticks(fontsize=AXIS_FONT_SIZE)
plt.yticks(fontsize=AXIS_FONT_SIZE)

# plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel(r'$m$',fontsize = FONT_SIZE, labelpad = 10)
plt.ylabel('Number of function evaluations', fontsize = FONT_SIZE, labelpad = 10)
# plt.title(r'Random Graph $G(n,m)$, $n\colon=2\sqrt{m}$',fontsize = FONT_SIZE)
# plt.title(r'Worst-Case Graph $G_{3,l}$',fontsize = FONT_SIZE)

plt.title(f'{GRAPH}',fontsize = FONT_SIZE)
plt.legend(loc='upper left', fontsize = FONT_SIZE)
# plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better visibility
plt.tight_layout()
plt.savefig(output_dir+f'\\mutation_{GRAPH}.png')
plt.show()
# %%

# %%
