
#%%
import numpy as np
import networkx as nx
import time
from pathlib import Path
import json
import cProfile
import math
import random
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import timeit
matplotlib.use("TkAgg")

def special_graph(h, l):
    G = nx.Graph()

    # Create nodes
    for i in range(1, h+1):
        for j in range(l+1):
            G.add_node((i, j))

    # Add horizontal edges
    for i in range(1, h+1):
        for j in range(0, l, 2):
            G.add_edge((i, j), (i, j+1))

    # Add complete bipartite graphs
    for j in range(1, l, 2):
        for i in range(1, h+1):
            for k in range(1, h+1):
                G.add_edge((i, j), (k, j+1))

    return G 


def special_graph(h, l):
    G = nx.Graph()

    # Create nodes
    for i in range(1, h+1):
        for j in range(l+1):
            G.add_node((i, j))

    # Add horizontal edges
    for i in range(1, h+1):
        for j in range(0, l, 2):
            G.add_edge((i, j), (i, j+1))

    # Add complete bipartite graphs
    for j in range(1, l, 2):
        for i in range(1, h+1):
            for k in range(1, h+1):
                G.add_edge((i, j), (k, j+1))

    return G

def get_graph_class(graph_class, graph_size, edge_creation_probability=0.5):

    if graph_class == "Path":
        graph = nx.path_graph(graph_size)
    elif graph_class == "Complete":
        graph = nx.complete_graph(graph_size)
    elif graph_class == "Cycle":
        graph = nx.cycle_graph(graph_size)
    elif graph_class == "Grid":
        graph = nx.grid_graph(graph_size)
    elif graph_class == "Torus":
        graph = nx.grid_2d_graph(graph_size[0], graph_size[1], periodic=True)
    elif graph_class == "Binary Tree":
        graph = nx.balanced_tree(2, graph_size)
    elif graph_class == "Random":
        graph = nx.gnm_random_graph(graph_size[0], graph_size[1], seed=42)
    elif graph_class == "Special":
        graph = special_graph(graph_size[0], graph_size[1])
    elif graph_class == "Complete Bipartite":
        graph = nx.complete_bipartite_graph(graph_size[0], graph_size[1], create_using=None)
    else:
        raise ValueError("Unknown specified graph.")
    return graph

def generate_random_solution(adjacency_matrix):

    non_nan_positions = np.column_stack(np.where(~np.isnan(adjacency_matrix)))
    colors = np.random.randint(0, 2, size=adjacency_matrix.shape[0])
    colored_matrix = adjacency_matrix.copy()
    colored_matrix[non_nan_positions[:, 0], non_nan_positions[:, 1]] = colors[non_nan_positions[:, 0]]
    colored_matrix[non_nan_positions[:, 1], non_nan_positions[:, 0]] = colors[non_nan_positions[:, 1]]
    if np.random.random() <= 0.5:
        colored_matrix = np.triu(colored_matrix) + np.triu(colored_matrix, 1).T
    else:
        colored_matrix = np.tril(colored_matrix) + np.tril(colored_matrix, -1).T

    return colored_matrix

def evaluate_fitness(adjacency_matrix, num_edges):
    row_sums = np.nansum(adjacency_matrix, axis=1)
    penalty = np.sum(np.maximum(0,row_sums-1)*(num_edges+1))
    matching_count = np.sum(row_sums)/2
    fitness_value = matching_count - penalty
    return fitness_value

def discrete_power_law_distribution(n, beta):

    interval_min = 1
    interval_max = math.floor(n/2)
    sample_values = np.arange(interval_min, interval_max+1) # values to sample
    zeta_beta = np.sum(1/np.power(sample_values, beta)) # cache Riemann zeta function value
    probabilities = 1/np.power(sample_values, beta) / zeta_beta
    return sample_values, probabilities


def calculate_probability_matrix_for_biased_mutation(adjacency_matrix, num_edges):
    adjacency_matrix = np.asarray(adjacency_matrix)
    column_sums = np.asarray(np.nansum(adjacency_matrix, axis=0))
    row_sums = np.asarray(np.nansum(adjacency_matrix, axis=1))

    deg_u = row_sums  # Convert to column vector
    deg_v = column_sums[None].T

    edge_weight = np.full(adjacency_matrix.shape, np.nan)

    condition_1 = (deg_u < 1) & (deg_v < 1) & ((adjacency_matrix == 1) | (adjacency_matrix == 0))
    condition_2 = (deg_u > 1) & (deg_v > 1) & (adjacency_matrix == 1)
    condition_3 = ((deg_u < 1) & (deg_v == 1)) | ((deg_u == 1) & (deg_v < 1)) & ((adjacency_matrix == 1) | (adjacency_matrix == 0))
    condition_4 = (((deg_u == 1) & (deg_v > 1)) | ((deg_u > 1) & (deg_v == 1))) & (adjacency_matrix == 1)
    condition_5 = (((deg_u > 1) & (deg_v < 1)) | ((deg_u < 1) & (deg_v > 1))) & ((adjacency_matrix == 1) | (adjacency_matrix == 0))
    condition = condition_1 | condition_2 | condition_3 | condition_4
    weight_matrix = np.where(condition, 1, 0)

    nan_positions = np.transpose(np.where(np.isnan(adjacency_matrix)))
    weight_matrix = weight_matrix.astype(float)
    # Fill the specified positions with NaN values
    weight_matrix[nan_positions[:, 0], nan_positions[:, 1]] = np.nan

    total_weight = np.nansum(weight_matrix) // 2
    if total_weight != 0:
        probabilities = (weight_matrix / (2 * total_weight)) + (1 / num_edges)
        # probabilities = (weight_matrix/2.5) + (1 / num_edges)
    else:
        print("error")
        return None
    return probabilities

def find_valid_neighbors(adj_matrix, current_vertex):
    adj_matrix = np.asmatrix(adj_matrix)
    return np.where(np.logical_or(adj_matrix[current_vertex] == 0, adj_matrix[current_vertex] == 1))[1]

def flip_edge(matrix, vertex1, vertex2):
    matrix[vertex1, vertex2] = 1 - matrix[vertex1, vertex2]
    matrix[vertex2, vertex1] = 1 - matrix[vertex2, vertex1]

def get_starting_vertex(matrix):
    flattened_matrix = matrix.flatten()
    probabilities = flattened_matrix / np.nansum(flattened_matrix)
    probabilities = np.nan_to_num(probabilities, nan=0.0)
    indices = list(range(len(flattened_matrix)))
    # sampled_index = np.random.choice(indices, p=probabilities)
    sampled_index = random.choices(indices, list(probabilities), k=1)[0]
    sampled_row = sampled_index // matrix.shape[1]
    sampled_col = sampled_index % matrix.shape[1]
    starting_vertex = random.choice([sampled_row, sampled_col])
    return starting_vertex


def one_plus_one_ea(
    graph,
    max_matching_deterministic,
    mutation_operator_name,
    beta_value=None
    ):
    adjacency_matrix = np.asmatrix(nx.to_numpy_array(graph))
    adjacency_matrix[adjacency_matrix == 0] = np.nan
    num_edges = np.count_nonzero(~np.isnan(adjacency_matrix)) // 2

    best_solution = generate_random_solution(adjacency_matrix)
    best_fitness = evaluate_fitness(best_solution, num_edges)
    max_fitness = len(max_matching_deterministic)

    steps = 1
    steps_previous = 0
    fitness_steps = {}
    fitness_steps[best_fitness] = steps
    previous_fitness_value = best_fitness
    convergence = {}

    def standard_mutation(adjacency_matrix, num_edges):
        probability = 1 / num_edges
        random_matrix = np.random.random(adjacency_matrix.shape)
        flip_mask = random_matrix < probability
        new_matrix = np.where(flip_mask, 1 - adjacency_matrix, adjacency_matrix)
        
        if np.random.random() <= 0.5:
            new_matrix = np.triu(new_matrix) + np.triu(new_matrix, 1).T
        else:
            new_matrix = np.tril(new_matrix) + np.tril(new_matrix, -1).T
        
        return new_matrix
    
    def heavy_tailed_mutation(adjacency_matrix, sample_values, probabilities, num_edges):
        probability_matrix = np.random.choice(sample_values, size=adjacency_matrix.shape, replace=True, p=probabilities)
        probability_matrix = probability_matrix / num_edges
        random_matrix = np.random.random(probability_matrix.shape)
        flip_mask = random_matrix < probability_matrix
        new_matrix = np.where(flip_mask, 1 - adjacency_matrix, adjacency_matrix)
        p =  np.random.random()
        if p <= 0.5:
            new_matrix = np.triu(new_matrix) + np.triu(new_matrix, 1).T
        else:
            new_matrix = np.tril(new_matrix) + np.tril(new_matrix, -1).T
        new_matrix = np.matrix(new_matrix)
        return new_matrix
    
    def biased_mutation(adjacency_matrix, num_edges):
        probabilities = calculate_probability_matrix_for_biased_mutation(adjacency_matrix, num_edges)
        random_matrix = np.random.random(probabilities.shape)
        flip_mask = random_matrix < probabilities
        new_matrix = np.where(flip_mask, 1 - adjacency_matrix, adjacency_matrix)
        p =  np.random.random()
        if p <= 0.5:
            new_matrix = np.triu(new_matrix) + np.triu(new_matrix, 1).T
        else:
            new_matrix = np.tril(new_matrix) + np.tril(new_matrix, -1).T
        new_matrix = np.matrix(new_matrix)
        return new_matrix
    
    def flip_random_path(adjacency_matrix, sample_values, probabilities, num_edges):
        
        path_length = np.random.choice(sample_values, size=1, p=probabilities)[0]
        start_vertex = np.random.choice(adjacency_matrix.shape[0])
        current_vertex = start_vertex
        path = [current_vertex]
        matrix_temp = adjacency_matrix.copy()

        for _ in range(path_length):
            valid_neighbors = find_valid_neighbors(matrix_temp, current_vertex)

            if len(valid_neighbors) == 0:
                break

            next_vertex = np.random.choice(valid_neighbors)
            path.append(next_vertex)
            matrix_temp[current_vertex, :] = np.nan
            matrix_temp[:, current_vertex] = np.nan
            current_vertex = next_vertex

        modified_matrix = adjacency_matrix.copy()

        for i in range(len(path) - 1):
            vertex1 = path[i]
            vertex2 = path[i + 1]
            flip_edge(modified_matrix, vertex1, vertex2)

        return modified_matrix

    def flip_simple_path(adjacency_matrix, num_edges):

        start_vertex = np.random.choice(adjacency_matrix.shape[0])
        current_vertex = start_vertex
        path = [current_vertex]
        matrix_temp = adjacency_matrix.copy()
        adj_matrix = np.asmatrix(adjacency_matrix)
        PREVIOUS_EDGE_MATCHING = None
        while len(path) < 3:
        # while True:
            if PREVIOUS_EDGE_MATCHING is None:
                valid_neighbors = find_valid_neighbors(matrix_temp, current_vertex)
                if len(valid_neighbors) == 0:
                    break
                else:
                    next_vertex = np.random.choice(valid_neighbors)
                    if matrix_temp[current_vertex, next_vertex] == 1:
                        PREVIOUS_EDGE_MATCHING = True
                    else:
                        PREVIOUS_EDGE_MATCHING = False
                    path.append(next_vertex)
                    matrix_temp[current_vertex, :] = np.nan
                    matrix_temp[:, current_vertex] = np.nan
                    current_vertex = next_vertex

            elif PREVIOUS_EDGE_MATCHING == False:
                valid_neighbors = np.where(matrix_temp[current_vertex] == 1)[0]
                if len(valid_neighbors) == 0:
                    break
                else:
                    next_vertex = np.random.choice(valid_neighbors)
                    path.append(next_vertex)
                    matrix_temp[current_vertex, :] = np.nan
                    matrix_temp[:, current_vertex] = np.nan
                    current_vertex = next_vertex
                PREVIOUS_EDGE_MATCHING = True
            elif PREVIOUS_EDGE_MATCHING == True:
                valid_neighbors = np.where(matrix_temp[current_vertex] == 0)[0]
                if len(valid_neighbors) == 0:
                    break
                else:
                    next_vertex = np.random.choice(valid_neighbors)
                    path.append(next_vertex)
                    matrix_temp[current_vertex, :] = np.nan
                    matrix_temp[:, current_vertex] = np.nan
                    current_vertex = next_vertex
                PREVIOUS_EDGE_MATCHING = False

        modified_matrix = adjacency_matrix.copy()
        for i in range(len(path) - 1):
            vertex1 = path[i]
            vertex2 = path[i + 1]
            flip_edge(modified_matrix, vertex1, vertex2)

        return modified_matrix

    def biased_flip_random_path(adjacency_matrix, sample_values, probabilities, num_edges):
        probability_matrix = calculate_probability_matrix_for_biased_mutation(adjacency_matrix, num_edges)
        start_vertex = get_starting_vertex(probability_matrix)
        path_length = np.random.choice(sample_values, size=1, p=probabilities)[0]
        # start_vertex = np.random.choice(adjacency_matrix.shape[0])
        current_vertex = start_vertex
        path = [current_vertex]
        matrix_temp = adjacency_matrix.copy()

        for _ in range(path_length):
            valid_neighbors = find_valid_neighbors(matrix_temp, current_vertex)

            if len(valid_neighbors) == 0:
                break

            next_vertex = np.random.choice(valid_neighbors)
            path.append(next_vertex)
            matrix_temp[current_vertex, next_vertex] = np.nan
            matrix_temp[next_vertex, current_vertex] = np.nan
            current_vertex = next_vertex

        modified_matrix = adjacency_matrix.copy()

        for i in range(len(path) - 1):
            vertex1 = path[i]
            vertex2 = path[i + 1]
            flip_edge(modified_matrix, vertex1, vertex2)

        return modified_matrix

    mutation_operators = {
        "Standard Mutation": standard_mutation,
        "Heavy Tailed Mutation": heavy_tailed_mutation,
        "Biased Mutation": biased_mutation,
        "Flip Random Path Mutation": flip_random_path,
        "Biased Flip Random Path Mutation": biased_flip_random_path,
        "Flip Simple Path Mutation": flip_simple_path
    }
    mutation_operator = mutation_operators.get(mutation_operator_name)
    if mutation_operator is None:
        raise ValueError("Unknown Mutation Operator")
    elif mutation_operator_name == "Flip Random Path Mutation" or mutation_operator_name == "Biased Flip Random Path Mutation" or mutation_operator_name == "Heavy Tailed Mutation":
        sample_values, probabilities = discrete_power_law_distribution(num_edges, beta_value)

    while best_fitness < max_fitness:
        if mutation_operator_name == "Flip Random Path Mutation" or mutation_operator_name == "Biased Flip Random Path Mutation" or mutation_operator_name == "Heavy Tailed Mutation":
            new_solution = mutation_operator(best_solution, sample_values, probabilities, num_edges)
        else:
            new_solution = mutation_operator(best_solution, num_edges)
        new_fitness_value = evaluate_fitness(new_solution, num_edges)
        
        if new_fitness_value >= best_fitness:
            best_fitness = new_fitness_value
            best_solution = new_solution
        
        steps += 1
        if best_fitness not in fitness_steps:
            if best_fitness > previous_fitness_value:
                fitness_steps[previous_fitness_value] = steps - steps_previous
                
                convergence[previous_fitness_value] = steps
                
                steps_previous = steps
                previous_fitness_value = best_fitness

    return best_solution, best_fitness, steps, fitness_steps, convergence

def visualize_graph(best_solution):
    
    adjacency_matrix = np.where(best_solution == 0, 1, np.where(np.isnan(best_solution), 0, best_solution))
    G = nx.Graph(adjacency_matrix)
    graph = G.copy()
    edges_ones = []
    edges_zeros = []

    for i in range(best_solution.shape[0]):

        for j in range(i+1, best_solution.shape[1]):
            value = best_solution[i, j]
            
            if np.isnan(value):
                continue
            elif value == 1:
                edges_ones.append((i, j))
            elif value == 0:
                edges_zeros.append((i, j))

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    nx.draw_networkx_edges(graph, pos, edgelist=edges_zeros, width=1.5, alpha=0.5, edge_color="gray")
    nx.draw_networkx_edges(graph, pos, edgelist=edges_ones, width=1.5, alpha=0.5, edge_color="red")
    plt.show()

def map_fitness_levels(data_list):
    combined_data = defaultdict(list)

    for data_dict in data_list:
        for x_val, y_val in data_dict.items():
            combined_data[x_val].append(y_val)

    x_values = []
    y_values = []
    max_fitness_value = np.max(list(combined_data.keys()))
    threshold = -20
    for x_val, y_vals in combined_data.items():
        if x_val >= threshold:
            x_values.append(x_val)
            avg_y = sum(y_vals) / len(y_vals)
            y_values.append(avg_y)

    x_values_sorted = sorted(x_values)
    y_values_sorted = [y for _, y in sorted(zip(x_values, y_values))]
    return x_values_sorted, y_values_sorted
    
def measure_max_weight_matching(graph):
    result = nx.max_weight_matching(graph, maxcardinality=False)
def run_experiment(
    graph_name,
    graph_size,
    mutation_operator_name,
    run_algorithm_count,
    experiment_name,
    beta_value = None
):  
    base_path = Path.cwd()
    # result_path: Path = base_path / f"experiment_results\{experiment_name}.json"
    results_directory_path = base_path/ f"experiment_results\{experiment_name}"
    if not results_directory_path.exists():
        results_directory_path.mkdir(parents=True, exist_ok=True)
        print("Directory created.")
    # else:
    #     raise ValueError("Directory already exists.")

    graph = get_graph_class(graph_name, graph_size)
    graph_edges_size = len(graph.edges)
    total_steps = []
    total_convergence = []
    total_fitness_steps = []
    start_time = time.time()
    print(f"Running experiment for {graph_name} of size {graph_edges_size} with {mutation_operator_name}:")
    print("==========================================================")
    for i in range(run_algorithm_count):
        max_matching_deterministic = nx.max_weight_matching(graph, maxcardinality=False)
        best_solution, best_fitness, steps, fitness_steps, convergence = one_plus_one_ea(graph, max_matching_deterministic, mutation_operator_name, beta_value=beta_value)
        total_steps.append(steps)
        total_convergence.append(convergence)
        total_fitness_steps.append(fitness_steps)
        print(f"Experiment {i}: {steps}")

    # visualize_fitness_levels(total_fitness_steps)
    # visualize_fitness_levels(total_convergence)
    average_optimization_time = sum(total_steps) / len(total_steps)
    end_time = time.time()
    execution_time = end_time - start_time
    if beta_value:
        experiment_parameters = [graph_name, graph_edges_size, mutation_operator_name,  beta_value, run_algorithm_count]
    else:
        experiment_parameters = [graph_name, graph_edges_size, mutation_operator_name, run_algorithm_count]
    result = {
        'RunTime': execution_time,
        'Parameters': experiment_parameters,
        'OptimizationSteps': total_steps,
        'AverageOptimizationSteps': average_optimization_time
    }
    # exp_name = f"{graph_name}_{graph_edges_size}_{mutation_operator_name}"
    exp_name_path = results_directory_path / f"{graph_name}_{graph_edges_size}_{mutation_operator_name}_{beta_value}.json"
    with open(exp_name_path, 'w') as json_file:
        json.dump(result, json_file)

    print(f"Average optimization time: {average_optimization_time}")
    print(f"Runtime in seconds: {execution_time}")
    print("==========================================================")
    return best_solution, best_fitness, steps, total_fitness_steps


def run_experiments_for_binary_tree():
    graph_name = "Binary Tree"
    graph_sizes = [3,4,5,6,7,8]
    mutation_operators = [
        "Standard Mutation",
        "Heavy Tailed Mutation",
        "Biased Mutation",
        "Flip Random Path Mutation",
        "Biased Flip Random Path Mutation"
    ]
    beta_value = 2
    run_algorithm_count = 50
    experiment_name = "Tree 2"

    for graph_size in graph_sizes:
        for mutation_operator_name in mutation_operators:
            (best_solution, best_fitness, steps) = run_experiment(
                graph_name,
                graph_size,
                mutation_operator_name,
                run_algorithm_count,
                experiment_name,
                beta_value
            )
    return None

def run_experiments_for_complete_bipartite_graph():
    graph_name = "Complete Bipartite"
    graph_sizes = [[20,20],[25,25],[30,30],[35,35],[40,40],[45,45],[50,50]]
    mutation_operators = [
        "Standard Mutation",
        "Heavy Tailed Mutation",
        "Biased Mutation",
        "Flip Random Path Mutation",
        "Biased Flip Random Path Mutation"
    ]
    beta_value = 2
    run_algorithm_count = 50
    experiment_name = "Path"

    for graph_size in graph_sizes:
        for mutation_operator_name in mutation_operators:
            (best_solution, best_fitness, steps) = run_experiment(
                graph_name,
                graph_size,
                mutation_operator_name,
                run_algorithm_count,
                experiment_name,
                beta_value
            )
    return None


def run_experiments_for_random_graph():
    graph_name = "Random"
    edge_sizes = [30,60,90,120,150]
    mutation_operators = [
        "Standard Mutation",
        "Heavy Tailed Mutation",
        "Biased Mutation",
        "Flip Random Path Mutation",
        "Biased Flip Random Path Mutation"
    ]
    beta_value = 2
    run_algorithm_count = 50
    experiment_name = "Random Non-Sparse"

    for m in edge_sizes:
        # n = round(2*(math.sqrt((2*m)+1)-1))
        n = round(2*math.sqrt(m))
        graph_size = [n,m]
        for mutation_operator_name in mutation_operators:
            (best_solution, best_fitness, steps) = run_experiment(
                graph_name,
                graph_size,
                mutation_operator_name,
                run_algorithm_count,
                experiment_name,
                beta_value
            )
    return None

def run_experiments_for_special_graph():
    graph_name = "Special"
    graph_sizes = [[3,15]]
    mutation_operators = [
        "Flip Random Path Mutation"
    ]
    beta_value = 2
    run_algorithm_count = 50
    experiment_name = "Special"

    for graph_size in graph_sizes:
        for mutation_operator_name in mutation_operators:
            (best_solution, best_fitness, steps) = run_experiment(
                graph_name,
                graph_size,
                mutation_operator_name,
                run_algorithm_count,
                experiment_name,
                beta_value
            )
    return None

def run_experiments_for_flip_random_path_mutation():
    graph_name = "Grid"
    graph_sizes = [[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11]]
    mutation_operators = [
        "Flip Random Path Mutation",
        "Biased Flip Random Path Mutation"
    ]
    beta_values = [2]
    run_algorithm_count = 50
    experiment_name = "Grid Analysis 2"

    for graph_size in graph_sizes:
        for mutation_operator_name in mutation_operators:
            for beta_value in beta_values:
                (best_solution, best_fitness, steps, total_fitness_steps) = run_experiment(
                    graph_name,
                    graph_size,
                    mutation_operator_name,
                    run_algorithm_count,
                    experiment_name,
                    beta_value
                )
    return None
def debug():
    graph_name = "Complete Bipartite"
    graph_sizes = [[20,20]]
    mutation_operators = [
        "Standard Mutation",
        "Heavy Tailed Mutation",
        "Biased Mutation",
        "Flip Random Path Mutation",
        "Biased Flip Random Path Mutation"
    ]
    beta_value = 2
    run_algorithm_count = 50
    experiment_name = "debug complete bipartite"

    for graph_size in graph_sizes:
        for mutation_operator_name in mutation_operators:
            (best_solution, best_fitness, steps, _) = run_experiment(
                graph_name,
                graph_size,
                mutation_operator_name,
                run_algorithm_count,
                experiment_name,
                beta_value
            )
    return None

def run_experiments_for_path_odd():
    graph_name = "Grid"
    graph_sizes = [[5,5]]
    mutation_operators = [
        "Heavy Tailed Mutation",
        "Biased Mutation",
        "Flip Random Path Mutation",
        "Biased Flip Random Path Mutation"
    ]
    beta_value = 2
    run_algorithm_count = 50
    experiment_name = "Path Odd"

    for graph_size in graph_sizes:
        for mutation_operator_name in mutation_operators:
            (best_solution, best_fitness, steps,_) = run_experiment(
                graph_name,
                graph_size,
                mutation_operator_name,
                run_algorithm_count,
                experiment_name,
                beta_value
            )
    return None
#%%
# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
# run_experiments_for_binary_tree()
# run_experiments_for_torus_graph()
    # profiler.disable()
    # profiler.print_stats(sort='cumulative')
# run_experiments_for_flip_random_path_mutation()
# debug()
run_experiments_for_path_odd()
# run_fitness_level_visualization()
# run_experiments_for_special_graph()
# %%
# run_experiments_for_flip_simple_path()
#%%
