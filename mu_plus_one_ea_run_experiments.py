#%%
import numpy as np
import networkx as nx
import time
from pathlib import Path
import json
import cProfile
import math
import random

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
    elif graph_class == "Random Bipartite":
        graph = nx.bipartite.random_graph(graph_size[0], graph_size[1], edge_creation_probability,seed=42)
    else:
        raise ValueError("Unknown specified graph.")
    return graph

def generate_random_solution(adjacency_matrix):
    # num_nodes = adjacency_matrix.shape[0]
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

def discrete_power_law_distribution(n, beta, interval_minimum):

    interval_min = interval_minimum
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
    weight_matrix = np.where(condition, 2, 0)

    nan_positions = np.transpose(np.where(np.isnan(adjacency_matrix)))
    weight_matrix = weight_matrix.astype(float)
    # Fill the specified positions with NaN values
    weight_matrix[nan_positions[:, 0], nan_positions[:, 1]] = np.nan

    total_weight = np.nansum(weight_matrix) // 2
    if total_weight != 0:
        probabilities = (weight_matrix / (2 * total_weight)) + (1 / num_edges)
    else:
        nan_mask = np.isnan(weight_matrix)
        probabilities = np.copy(weight_matrix)
        probabilities[~nan_mask] = 1/num_edges
    return probabilities

def find_valid_neighbors(adj_matrix, current_vertex):
    adj_matrix = np.asmatrix(adj_matrix)
    return np.where(np.logical_or(adj_matrix[current_vertex] == 0, adj_matrix[current_vertex] == 1))[1]

def flip_edge(matrix, vertex1, vertex2):
    matrix[vertex1, vertex2] = 1 - matrix[vertex1, vertex2]
    matrix[vertex2, vertex1] = 1 - matrix[vertex2, vertex1]

def get_starting_vertex(matrix):
    flattened_matrix = matrix.flatten()
    # probabilities = flattened_matrix / np.nansum(flattened_matrix)
    probabilities = flattened_matrix.copy()
    probabilities = np.nan_to_num(probabilities, nan=0.0)
    indices = list(range(len(flattened_matrix)))
    # sampled_index = np.random.choice(indices, p=probabilities)
    sampled_index = random.choices(indices, list(probabilities), k=1)[0]
    sampled_row = sampled_index // matrix.shape[1]
    sampled_col = sampled_index % matrix.shape[1]
    starting_vertex = random.choice([sampled_row, sampled_col])
    return starting_vertex

def parent_selection_mechanism(population_fitness_input):
    population_fitness = population_fitness_input.copy()
    min_value = min(population_fitness.values())
    for key, value in population_fitness.items():
        population_fitness[key] = value + abs(min_value)+1
    total = sum(abs(v) for v in population_fitness.values())
    probabilities = {k: v/total for k, v in population_fitness.items()}

    first_selection = random.choices(list(population_fitness.keys()), list(probabilities.values()), k=1)[0]
    remaining_keys = list(population_fitness.keys())
    if len(remaining_keys)<=1:
        second_selection = None
    else:
        remaining_keys.remove(first_selection)

        remaining_probabilities = {k: v/total for k, v in population_fitness.items() if k != first_selection}

        second_selection = random.choices(remaining_keys, list(remaining_probabilities.values()), k=1)[0]

    return [first_selection, second_selection]

def inverse_crossover_2(parent1, parent2, paths):
    # Create a mask for NaN values in either parent matrix

    nan_mask = np.isnan(parent1) | np.isnan(parent2)
    large_paths = [lst for lst in paths if len(lst) > 2]
    if len(large_paths)>1:
        selected_paths = random.sample(large_paths, int(len(large_paths)/2))
    else:
        selected_paths = random.sample(paths, int(len(paths)/2))
    # Create a child matrix with default value 1
    child = np.ones(parent1.shape, dtype=float)
    subset_mask = create_boolean_matrix(parent1, selected_paths)
    
    # Apply rules for 0 and 1 combinations
    child[(parent1 == 0) & (parent2 == 0)] = 1
    child[(parent1 == 1) & (parent2 == 1)] = 0
    
    # Apply rules for 0 and 1 combinations with probability 0.5
    probability_mask = (parent1 == 0) & (parent2 == 1)
    child[probability_mask] = np.random.choice([0, 1], size=np.count_nonzero(probability_mask), p=[0.5, 0.5])

    final_child = np.ones(parent1.shape, dtype=float)
    final_child[subset_mask] = child[subset_mask]
    child2 = 1-child
    final_child[~subset_mask] = child2[~subset_mask]
    
    # Apply mask to handle NaN combinations
    final_child[nan_mask] = np.nan
    
    return final_child, final_child


def uniform_crossover(parent1, parent2):
    rows, cols = parent1.shape
    crossover_mask = np.random.rand(rows, cols) < 0.5

    child1 = np.where(crossover_mask, parent2, parent1)
    child2 = np.where(crossover_mask, parent1, parent2)

    if np.random.random() <= 0.5:
        triu_mask = np.triu(np.ones((rows, cols), dtype=bool), k=1)
        child1 = np.where(triu_mask, child1, child1.T)
        child2 = np.where(triu_mask, child2, child2.T)
    else:
        tril_mask = np.tril(np.ones((rows, cols), dtype=bool), k=-1)
        child1 = np.where(tril_mask, child1, child1.T)
        child2 = np.where(tril_mask, child2, child2.T)

    return child1, child2

def k_point_crossover(parent1, parent2, k):
    if k >= parent1.shape[0] * parent1.shape[1]:
        raise ValueError("Invalid k value for k-point crossover")

    total_points = parent1.shape[0] * parent1.shape[1]
    crossover_points = np.sort(np.random.choice(total_points, k, replace=False))

    crossover_mask = np.zeros_like(parent1, dtype=bool)
    crossover_mask.flat[crossover_points] = True

    child1 = np.where(crossover_mask, parent2, parent1)
    child2 = np.where(crossover_mask, parent1, parent2)

    if np.random.random() <= 0.5:
        triu_mask = np.triu(np.ones_like(parent1, dtype=bool), k=1)
        child1 = np.where(triu_mask, child1, child1.T)
        child2 = np.where(triu_mask, child2, child2.T)
    else:
        tril_mask = np.tril(np.ones_like(parent1, dtype=bool), k=-1)
        child1 = np.where(tril_mask, child1, child1.T)
        child2 = np.where(tril_mask, child2, child2.T)

    return child1, child2

def crossover(parent1, parent2, crossover_type='Uniform', subset_nodes= None, k=None):
    if parent1.shape != parent2.shape:
        raise ValueError("Parent matrices must have the same dimensions")

    if crossover_type == 'Uniform':
        return uniform_crossover(parent1, parent2)
    elif crossover_type == "Inverse Probabilistic":
        # if subset_nodes is None:
        #     raise ValueError("No partition set is given")
        return inverse_crossover_probabilistic(parent1, parent2)
    elif crossover_type == "Inverse":
        return inverse_crossover(parent1, parent2)
    else:
        raise ValueError("Invalid crossover type")

def inverse_crossover(parent1, parent2):
    nan_mask = np.isnan(parent1) | np.isnan(parent2)
    
    # Create a child matrix with default value 1
    child = np.ones(parent1.shape, dtype=float)
    

    child[(parent1 == 0) & (parent2 == 0)] = 1
    child[(parent1 == 1) & (parent2 == 1)] = 0
    child[(parent1 == 0) & (parent2 == 1)] = np.random.choice([0,1])
    child[(parent1 == 1) & (parent2 == 0)] = np.random.choice([0,1])

    # Apply mask to handle NaN combinations
    child[nan_mask] = np.nan
    
    return child, child

def inverse_crossover_probabilistic(parent1, parent2):
    nan_mask = np.isnan(parent1) | np.isnan(parent2)
    
    # Create a child matrix with default value 1
    child = np.ones(parent1.shape, dtype=float)

    p = np.random.random()
    if p < 0.9:
        child[(parent1 == 0) & (parent2 == 0)] = 1
    else:
        child[(parent1 == 0) & (parent2 == 0)] = 1
    p = np.random.random()
    if p<0.9:
        child[(parent1 == 1) & (parent2 == 1)] = 0
    else:
        child[(parent1 == 1) & (parent2 == 1)] = 1
    child[(parent1 == 0) & (parent2 == 1)] = np.random.choice([0,1])
    child[(parent1 == 1) & (parent2 == 0)] = np.random.choice([0,1])
    
    # Apply mask to handle NaN combinations
    child[nan_mask] = np.nan
    
    return child, child

def find_random_path_2(
    graph,
    starting_edge,
    sample_values,
    probabilities,
    beta
    ):

    path = []
    n = graph.number_of_edges()
    # sample_values, probabilities = discrete_power_law_distribution(n,beta=beta)
    output = np.random.choice(sample_values, size=1, replace=True, p=probabilities)
    path_length = output[0]
    # path_length = random.randint(2,9)
    while len(path) < path_length:
        path.append(tuple(sorted(starting_edge)))
        graph.remove_edges_from([tuple(sorted(starting_edge))])
        current_vertex = starting_edge[1]
        incident_edges = list(graph.edges(current_vertex))
        if incident_edges:
            starting_edge = random.choice(incident_edges)
        else:
            break
    return path

def generate_random_paths(graph, sample_values, probabilites):
    free_edges = set(list(graph.edges()))

    paths = []

    while free_edges and len(graph.edges())>1:
        # Randomly select a free vertex
        start_edge = random.choice(list(free_edges))
        path = find_random_path_2(graph, start_edge,sample_values, probabilites, 1.01)
        if len(path) >0:
            paths.append(path)
            free_edges.difference_update(path)
    paths.append(list(free_edges))
    return paths

def create_boolean_matrix(matrix, path_list):

    boolean_matrix = np.zeros(matrix.shape, dtype=bool)
    for path in path_list:
        for edge in path:
            row = edge[0]
            col = edge[1]
            boolean_matrix[row][col] = True
            boolean_matrix[col][row] = True

    return boolean_matrix

def mu_plus_one_ea(
    graph,
    graph_name,
    max_matching_deterministic,
    crossover_name,
    parent_population_size,
    crossover_probability,
    diversity_preserving_mechanism
    ):

    if graph_name == "Grid" or graph_name == "Torus":
        graph = nx.convert_node_labels_to_integers(graph, first_label=0)

    adjacency_matrix = np.asmatrix(nx.to_numpy_array(graph))
    adjacency_matrix[adjacency_matrix == 0] = np.nan
    num_edges = np.count_nonzero(~np.isnan(adjacency_matrix)) // 2
    vertex_degrees = np.sum(~np.isnan(adjacency_matrix), axis=1)
    max_fitness = len(max_matching_deterministic)

    population = {}
    population_fitness = {}

    for i in range(parent_population_size):
        initial_individual = generate_random_solution(adjacency_matrix)
        initial_individual_fitness_value = evaluate_fitness(initial_individual, num_edges)
        population[i] = initial_individual
        population_fitness[i] = initial_individual_fitness_value
    
    steps = 0
    population_max_fitness = max(population_fitness.values())
    population_min_fitness = min(population_fitness.values())
    
    while population_max_fitness < max_fitness:
        p = np.random.random()
        if p <= crossover_probability:
            
            [parent_1_index, parent_2_index] = parent_selection_mechanism(population_fitness)
            parent_1 = population[parent_1_index]
            parent_2 = population[parent_2_index]
            # partition_1, partition_2 = kernighan_lin_bisection(graph)
            # subset_nodes = list(random.choice([partition_1, partition_2]))

            child_1, child_2 = crossover(parent_1, parent_2, crossover_name)
            child_1_fitness = evaluate_fitness(child_1, num_edges)
            child_2_fitness = evaluate_fitness(child_1, num_edges)
            if child_2_fitness < child_1_fitness:
                best_child = child_1
            elif child_2_fitness > child_1_fitness:
                best_child = child_2
            else:
                best_child = random.choice([child_1, child_2])
        else:
            [parent_1_index, _] = parent_selection_mechanism(population_fitness)
            best_child = population[parent_1_index]
        mutated_child = standard_mutation(best_child, num_edges)
        # mutated_child = biased_flip_random_path(best_child, mutation_sample_values, mutation_probabilities, num_edges)
        mutated_child_fitness = evaluate_fitness(mutated_child, num_edges)
        duplicate = False

        # if mutated_child_fitness >= population_max_fitness:
        if mutated_child_fitness >= population_min_fitness:
            if diversity_preserving_mechanism:
                for parent_index, parent in population.items():
                    if np.all(np.isclose(mutated_child , parent , equal_nan=True)):
                        duplicate = True
                        break
                    else:
                        duplicate = False

            if not duplicate:
                min_keys = [key for key, value in population_fitness.items() if value == population_min_fitness]
                key_to_remove = random.choice(min_keys)
                population[key_to_remove] = mutated_child
                population_fitness[key_to_remove] = mutated_child_fitness
        population_max_fitness = max(population_fitness.values())
        population_min_fitness = min(population_fitness.values())

        steps += 1
    max_keys = [key for key, value in population_fitness.items() if value == population_max_fitness]
    max_key = random.choice(max_keys)
    best_solution = population[max_key]
    best_fitness = population_fitness[max_key]

    return best_solution, best_fitness, steps

def standard_mutation(adjacency_matrix, num_edges):
    # rows, cols = adjacency_matrix.shape
    probability = 1 / num_edges
    random_matrix = np.random.random(adjacency_matrix.shape)
    flip_mask = random_matrix < probability
    new_matrix = np.where(flip_mask, 1 - adjacency_matrix, adjacency_matrix)
    
    if np.random.random() <= 0.5:
        new_matrix = np.triu(new_matrix) + np.triu(new_matrix, 1).T
    else:
        new_matrix = np.tril(new_matrix) + np.tril(new_matrix, -1).T
    
    return new_matrix

def biased_flip_random_path(adjacency_matrix, sample_values, probabilities, num_edges):
        probability_matrix = calculate_probability_matrix_for_biased_mutation(adjacency_matrix, num_edges)
        start_vertex = get_starting_vertex(probability_matrix)
        path_length = np.random.choice(sample_values, size=1, p=probabilities)[0]
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

def run_experiment(
    graph_name,
    graph_size,
    run_algorithm_count,
    experiment_name,
    parent_population_size,
    crossover_name,
    crossover_probability,
    diversity_preserving_mechanism
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
    start_time = time.time()
    print(f"Running experiment for {graph_name} of size {graph_edges_size} with {crossover_name}, population:{parent_population_size}, div:{diversity_preserving_mechanism}:")
    print("==========================================================")
    for i in range(run_algorithm_count):
        max_matching_deterministic = nx.max_weight_matching(graph, maxcardinality=False)
        best_solution, best_fitness, steps = mu_plus_one_ea(
            graph,
            graph_name,
            max_matching_deterministic,
            crossover_name,
            parent_population_size,
            crossover_probability,
            diversity_preserving_mechanism
        )
        total_steps.append(steps)
        print(f"Experiment {i}: {steps}")
    average_optimization_time = sum(total_steps) / len(total_steps)
    end_time = time.time()
    execution_time = end_time - start_time
    experiment_parameters = [graph_name, graph_edges_size, crossover_name, run_algorithm_count, run_algorithm_count, parent_population_size, diversity_preserving_mechanism]
    result = {
        'RunTime': execution_time,
        'Parameters': experiment_parameters,
        'OptimizationSteps': total_steps,
        'AverageOptimizationSteps': average_optimization_time
    }
    # exp_name = f"{graph_name}_{graph_edges_size}_{mutation_operator_name}"
    exp_name_path = results_directory_path / f"{graph_name}_{graph_edges_size}_{crossover_name}_{parent_population_size}_{diversity_preserving_mechanism}.json"
    with open(exp_name_path, 'w') as json_file:
        json.dump(result, json_file)

    print(f"Average optimization time: {average_optimization_time}")
    print(f"Runtime in seconds: {execution_time}")
    print("==========================================================")
    return best_solution, best_fitness, steps

def debug():
    graph_name = "Path"
    graph_size = 31
    beta_value = 1.01
    run_algorithm_count = 50
    experiment_name = "debug"
    parent_population_size = 2
    crossover_name = "Inverse"
    divergence_preserving_mechanism = True
    crossover_probabilty = 0.0

    (best_solution, best_fitness, steps) = run_experiment(
        graph_name,
        graph_size,
        run_algorithm_count,
        experiment_name,
        parent_population_size,
        crossover_name,
        crossover_probabilty,
        beta_value,
        divergence_preserving_mechanism
    )
    return None

def run_experiments_for_tree():
    graph_name = "Binary Tree"
    graph_sizes = [3,4,5,6,7]
    run_algorithm_count = 50
    experiment_name = "Mu Plus One GA Tree 2"
    parent_population_sizes = [2,5,10]
    crossover_names = ["Uniform","Inverse"]
    divergence_preserving_mechanisms = [True, False]
    crossover_probabilty = 0.1
    for graph_size in graph_sizes:
        for crossover_name in crossover_names:
            for parent_population_size in parent_population_sizes:
                for divergence_preserving_mechanism in divergence_preserving_mechanisms:
                    if divergence_preserving_mechanism == True and crossover_name == "Inverse":
                        continue
                    else:
                        (best_solution, best_fitness, steps) = run_experiment(
                            graph_name,
                            graph_size,
                            run_algorithm_count,
                            experiment_name,
                            parent_population_size,
                            crossover_name,
                            crossover_probabilty,
                            divergence_preserving_mechanism
                        )
    return None

def run_experiments_for_toroid():
    graph_name = "Torus"
    graph_sizes = [[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]
    run_algorithm_count = 50
    experiment_name = "Mu Plus One GA Torus 2"
    parent_population_sizes = [2,5,10]
    crossover_names = ["Uniform","Inverse"]
    divergence_preserving_mechanisms = [True, False]
    crossover_probabilty = 0.1
    for graph_size in graph_sizes:
        for crossover_name in crossover_names:
            for parent_population_size in parent_population_sizes:
                for divergence_preserving_mechanism in divergence_preserving_mechanisms:
                    if divergence_preserving_mechanism == True and crossover_name == "Inverse":
                        continue
                    else:
                        (best_solution, best_fitness, steps) = run_experiment(
                            graph_name,
                            graph_size,
                            run_algorithm_count,
                            experiment_name,
                            parent_population_size,
                            crossover_name,
                            crossover_probabilty,
                            divergence_preserving_mechanism
                        )
    return None

#%%
def run_experiments_for_complete_bipartite():
    graph_name = "Complete Bipartite"
    graph_sizes = [[20,20],[25,25],[30,30],[35,35],[40,40],[45,45]]
    run_algorithm_count = 50
    experiment_name = "Mu Plus One GA Complete Bipartite"
    parent_population_sizes = [2,5,10]
    crossover_names = ["Uniform","Inverse"]
    divergence_preserving_mechanisms = [True, False]
    crossover_probabilty = 0.1
    for graph_size in graph_sizes:
        for crossover_name in crossover_names:
            for parent_population_size in parent_population_sizes:
                for divergence_preserving_mechanism in divergence_preserving_mechanisms:
                    if divergence_preserving_mechanism == True and crossover_name == "Inverse":
                        continue
                    else:
                        (best_solution, best_fitness, steps) = run_experiment(
                            graph_name,
                            graph_size,
                            run_algorithm_count,
                            experiment_name,
                            parent_population_size,
                            crossover_name,
                            crossover_probabilty,
                            divergence_preserving_mechanism
                        )
    return None

#%%
def run_experiments_for_special_graph():
    graph_name = "Special"
    graph_sizes = [[3,3],[3,5],[3,7],[3,9],[3,11],[3,13],[3,15]]
    run_algorithm_count = 50
    experiment_name = "Mu Plus One GA Special"
    parent_population_sizes = [2,5,10]
    crossover_names = ["Uniform","Inverse"]
    divergence_preserving_mechanisms = [True, False]
    crossover_probabilty = 0.1
    for graph_size in graph_sizes:
        for crossover_name in crossover_names:
            for parent_population_size in parent_population_sizes:
                for divergence_preserving_mechanism in divergence_preserving_mechanisms:
                    if divergence_preserving_mechanism == True and crossover_name == "Inverse":
                        continue
                    else:
                        (best_solution, best_fitness, steps) = run_experiment(
                            graph_name,
                            graph_size,
                            run_algorithm_count,
                            experiment_name,
                            parent_population_size,
                            crossover_name,
                            crossover_probabilty,
                            divergence_preserving_mechanism
                        )
    return None
#%%
# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
    # debug()
# run_experiments_for_binary_tree()
    # profiler.disable()
    # profiler.print_stats(sort='cumulative')
run_experiments_for_tree()


# %%
