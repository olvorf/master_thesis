#%%
import json
from pathlib import Path

# Specify the directory where your JSON files are located
directory_path = Path(r'C:\Users\OV\Desktop\Uni\Master_Thesis_Project\experiment_results\Mu Plus One GA Tree 2')
GRAPH = "Complete Binary Tree"
FONT_SIZE = 20
AXIS_FONT_SIZE = 19
output_dir = r"C:\Users\OV\Desktop\Uni\Master_Thesis_Project\result_analysis\crossover_result_images_2"
# Create a list to store the data from the JSON files
data_list = []

# Use a loop to iterate through the files in the directory and read each JSON file
for file_path in directory_path.glob('*.json'):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        data_list.append(data)

#%%
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
    avg_optimization_steps = exp['AverageOptimizationSteps']
    div_mechanism = exp['Parameters'][6]
    if crossover_operator == "Uniform":
        if div_mechanism == True:
            if population == 2:
                uni_2_true[graph_size] = avg_optimization_steps
            elif population == 5:
                uni_5_true[graph_size] = avg_optimization_steps
            elif population == 10:
                uni_10_true[graph_size] = avg_optimization_steps
        elif div_mechanism == False:
            if population == 2:
                uni_2[graph_size] = avg_optimization_steps
            elif population == 5:
                uni_5[graph_size] = avg_optimization_steps
            elif population == 10:
                uni_10[graph_size] = avg_optimization_steps
    elif crossover_operator == "Inverse":
        if population == 2:
            inv_2[graph_size] = avg_optimization_steps
        elif population == 5:
            inv_5[graph_size] = avg_optimization_steps
        elif population == 10:
            inv_10[graph_size] = avg_optimization_steps

#%%
import matplotlib.pyplot as plt
x = list(inv_2.keys())  # Assuming all dictionaries have the same keys

x = sorted(x)
inv_2_values = [inv_2[key] for key in x]
inv_5_values = [inv_5[key] for key in x]
inv_10_values = [inv_10[key] for key in x]

uni_2_values = [uni_2[key] for key in x]
uni_5_values = [uni_5[key] for key in x]
uni_10_values = [uni_10[key] for key in x]

uni_2_true_values = [uni_2_true[key] for key in x]
uni_5_true_values = [uni_5_true[key] for key in x]
uni_10_true_values = [uni_10_true[key] for key in x]

plt.figure(figsize=(8, 5),dpi=350)
plt.plot(x, inv_2_values, linestyle='-',label='inverse', marker='o')
plt.plot(x, uni_2_values, linestyle='-',label='uniform', marker='s')
plt.plot(x, uni_2_true_values, linestyle='-',label='uniform with div.', marker='^')
plt.grid(True)
plt.xticks(fontsize=AXIS_FONT_SIZE)
plt.yticks(fontsize=AXIS_FONT_SIZE)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel(r'$m$',fontsize = FONT_SIZE, labelpad = 10)
plt.ylabel('Number of function evaluations', fontsize = FONT_SIZE, labelpad = 10)
plt.title(f'{GRAPH}'+', ' +r'$\mu=2$',fontsize = FONT_SIZE)
# plt.title(r'$\mu=2$',fontsize = 14)

plt.legend(loc='upper left', fontsize = FONT_SIZE)
plt.tight_layout()
plt.savefig(output_dir+f'\\crossover_{GRAPH}_2.png')
plt.show()

#%%
plt.figure(figsize=(8, 5),dpi=350)  # Optional: Set the figure size
plt.plot(x, inv_5_values, linestyle='-',label='inverse', marker='o')
plt.plot(x, uni_5_values, linestyle='-',label='uniform', marker='s')
plt.plot(x, uni_5_true_values, linestyle='-',label='uniform with div.', marker='^')
plt.grid(True)
plt.xticks(fontsize=AXIS_FONT_SIZE)
plt.yticks(fontsize=AXIS_FONT_SIZE)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel(r'$m$',fontsize = FONT_SIZE, labelpad = 10)
plt.ylabel('Number of function evaluations', fontsize = FONT_SIZE, labelpad = 10)
plt.title(f'{GRAPH}'+', ' +r'$\mu=5$',fontsize = FONT_SIZE)
# plt.title(r'$\mu=5$',fontsize = 14)
plt.legend(loc='upper left', fontsize = FONT_SIZE)
plt.tight_layout()
plt.savefig(output_dir+f'\\crossover_{GRAPH}_5.png')
plt.show()

#%%
plt.figure(figsize=(8, 5),dpi=350)  # Optional: Set the figure size
plt.plot(x, inv_10_values, linestyle='-',label='inverse', marker='o')
plt.plot(x, uni_10_values, linestyle='-',label='uniform', marker='s')
plt.plot(x, uni_10_true_values, linestyle='-',label='uniform with div.', marker='^')
plt.grid(True)
plt.xticks(fontsize=AXIS_FONT_SIZE)
plt.yticks(fontsize=AXIS_FONT_SIZE)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel(r'$m$',fontsize = FONT_SIZE, labelpad = 10)
plt.ylabel('Number of function evaluations', fontsize = FONT_SIZE, labelpad = 10)
plt.title(f'{GRAPH}'+', ' +r'$\mu=10$',fontsize = FONT_SIZE)
# plt.title(r'$\mu=10$',fontsize = 14)
plt.legend(loc='upper left', fontsize = FONT_SIZE)
plt.tight_layout()
plt.savefig(output_dir+f'\\crossover_{GRAPH}_10.png')
plt.show()

#%%