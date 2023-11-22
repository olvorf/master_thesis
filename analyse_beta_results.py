#%%
import json
from pathlib import Path

# Specify the directory where your JSON files are located
directory_path = Path(r'C:\Users\OV\Desktop\Uni\Master_Thesis_Project\experiment_results\Path Analysis 2')
GRAPH = "Path (even)"
FONT_SIZE = 17
AXIS_FONT_SIZE = 16
output_dir = r"C:\Users\OV\Desktop\Uni\Master_Thesis_Project\result_analysis\beta_analysis"
# Create a list to store the data from the JSON files
data_list = []

# Use a loop to iterate through the files in the directory and read each JSON file
for file_path in directory_path.glob('*.json'):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        data_list.append(data)

#%%
biased_beta_1 = {}
biased_beta_2 = {}
biased_beta_3 = {}
biased_beta_4 = {}
biased_beta_5 = {}
biased_beta_6 = {}
flip_beta_1 = {}
flip_beta_2 = {}
flip_beta_3 = {}
flip_beta_4 = {}
flip_beta_5 = {}
flip_beta_6 = {}
for exp in data_list:
    graph_size = exp['Parameters'][1]
    mutation_operator = exp['Parameters'][2]
    beta = exp['Parameters'][3]
    avg_optimization_steps = exp['AverageOptimizationSteps']
    if mutation_operator == "Biased Flip Random Path Mutation":
        if beta == 1.001:
            biased_beta_1[graph_size] = avg_optimization_steps
        elif beta == 1.5:
            biased_beta_2[graph_size] = avg_optimization_steps
        elif beta == 2:
            biased_beta_3[graph_size] = avg_optimization_steps
        elif beta == 2.5:
            biased_beta_4[graph_size] = avg_optimization_steps
        elif beta == 3:
            biased_beta_5[graph_size] = avg_optimization_steps
        elif beta == 3.5:
            biased_beta_6[graph_size] = avg_optimization_steps
    elif mutation_operator == "Flip Random Path Mutation":
        if beta == 1.001:
            flip_beta_1[graph_size] = avg_optimization_steps
        elif beta == 1.5:
            flip_beta_2[graph_size] = avg_optimization_steps
        elif beta == 2:
            flip_beta_3[graph_size] = avg_optimization_steps
        elif beta == 2.5:
            flip_beta_4[graph_size] = avg_optimization_steps
        elif beta == 3:
            flip_beta_5[graph_size] = avg_optimization_steps
        elif beta == 3.5:
            flip_beta_6[graph_size] = avg_optimization_steps
#%%
import matplotlib.pyplot as plt
x = list(biased_beta_1.keys())  # Assuming all dictionaries have the same keys

x = sorted(x)

beta_1_values = [biased_beta_1[key] for key in x]
beta_2_values = [biased_beta_2[key] for key in x]
beta_3_values = [biased_beta_3[key] for key in x]
beta_4_values = [biased_beta_4[key] for key in x]
beta_5_values = [biased_beta_5[key] for key in x]
beta_6_values = [biased_beta_6[key] for key in x]

plt.figure(figsize=(8, 5),dpi=350)  # Optional: Set the figure size

plt.plot(x, beta_1_values, linestyle='-',label='\u03B2=1.001', marker='o')
plt.plot(x, beta_2_values, linestyle='-',label='\u03B2=1.5', marker='s')
plt.plot(x, beta_3_values, linestyle='-',label='\u03B2=2', marker='^')
plt.plot(x, beta_4_values, linestyle='-',label='\u03B2=2.5', marker='x')
plt.plot(x, beta_5_values, linestyle='-', label='\u03B2=3', marker='D')
plt.plot(x, beta_6_values, linestyle='-', label='\u03B2=3.5', marker='+')
plt.grid(True)
plt.xticks(fontsize=AXIS_FONT_SIZE)
plt.yticks(fontsize=AXIS_FONT_SIZE)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel(r'$m$',fontsize = FONT_SIZE, labelpad = 10)
plt.ylabel('Number of function evaluations', fontsize = FONT_SIZE, labelpad = 10)
plt.title(GRAPH,fontsize = FONT_SIZE)
# plt.title(r'Worst-Case Graph $G_{3,l}$',fontsize = FONT_SIZE)
plt.legend(loc='upper left', fontsize = FONT_SIZE)
plt.tight_layout()  
plt.savefig(output_dir+f'\\beta_biased_{GRAPH}.png')
plt.show()
#%%
x = list(flip_beta_1.keys())  # Assuming all dictionaries have the same keys

x = sorted(x)

beta_1_values = [flip_beta_1[key] for key in x]
beta_2_values = [flip_beta_2[key] for key in x]
beta_3_values = [flip_beta_3[key] for key in x]
beta_4_values = [flip_beta_4[key] for key in x]
beta_5_values = [flip_beta_5[key] for key in x]
beta_6_values = [flip_beta_6[key] for key in x]

plt.figure(figsize=(8, 5),dpi=350)  # Optional: Set the figure size

plt.plot(x, beta_1_values, linestyle='-',label='\u03B2=1.001', marker='o')
plt.plot(x, beta_2_values, linestyle='-',label='\u03B2=1.5', marker='s')
plt.plot(x, beta_3_values, linestyle='-',label='\u03B2=2', marker='^')
plt.plot(x, beta_4_values, linestyle='-',label='\u03B2=2.5', marker='x')
plt.plot(x, beta_5_values, linestyle='-', label='\u03B2=3', marker='D')
plt.plot(x, beta_6_values, linestyle='-', label='\u03B2=3.5', marker='+')
plt.grid(True)
plt.xticks(fontsize=AXIS_FONT_SIZE)
plt.yticks(fontsize=AXIS_FONT_SIZE)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.xlabel(r'$m$',fontsize = FONT_SIZE, labelpad = 10)
plt.ylabel('Number of function evaluations', fontsize = FONT_SIZE, labelpad = 10)
plt.title(GRAPH,fontsize = FONT_SIZE)
# plt.title(r'Worst-Case Graph $G_{3,l}$',fontsize = FONT_SIZE)
plt.legend(loc='upper left', fontsize = FONT_SIZE)
plt.tight_layout()
plt.savefig(output_dir+f'\\beta_flip_{GRAPH}.png')
plt.show()
# %%
