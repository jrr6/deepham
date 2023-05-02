import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
from torch_geometric.data import Data
from typing import FrozenSet, Callable
from itertools import combinations
from tqdm import tqdm
from pathlib import Path
from random import randint

Edge = FrozenSet[int]


def generate_undirected_graph(edges: set[Edge]) -> list[list[int]]:
    """
    @param edges

    @return ensure the edges are doubled for an undirected graph
    """
    return [[x, y] for x, y in edges] + [[y, x] for x, y in edges]


def generate_random_edges(verts: int, num_rand_edges: int) -> tuple[set[Edge], set[FrozenSet[int]], int]:
    """
    Generates a random DIRECTED graph with (at least one) Hamiltonian Path and an arbitrary number of vertices

    @param verts:
        an integer indicating the number of vertices in this graph. Note the graph will be connected
        via its Hamiltonian path

    @param num_rand_edges:
        an integer indicating the number of random edges to create, in addition to the edges used
        to construct a Hamiltonian path

    @return:

    """

    path_seq = np.random.permutation(verts)
    path_edges = np.lib.stride_tricks.sliding_window_view(path_seq, 2)
    path_edges_set = set(map(frozenset, path_edges))

    possible_edges = np.array(list(combinations(path_seq, 2)))
    random_edge_indices = np.random.choice(len(list(combinations(path_seq, 2))), size=num_rand_edges)

    random_edges = set(map(frozenset, possible_edges[random_edge_indices, :]))

    return path_edges_set, path_edges_set.union(random_edges), path_edges[0, 0]


def generate_hampath_graph(verts: int, num_rand_edges) -> tuple[Data, int]:
    """
    Generates a random UNDIRECTED graph with the vertices {1, ..., verts} with
    a Hamiltonian path and *approximately* (verts - 1) + num_rand_edges many
    edges.

    @param verts:
        the number of vertices on which to construct the graph

    @param num_rand_edges:
        the approximate number of extra edges to insert into the graph in
        addition to the guaranteed Hamiltonian path

    @return:
        a torch_geometric Data object representing the generated graph
    """
    vertices = torch.arange(0, verts)
    vertices = F.one_hot(vertices, num_classes=verts).float()

    _, all_edges, starting_vertex = generate_random_edges(verts, num_rand_edges)
    edges_transposed = torch.tensor(np.array(list(map(list, generate_undirected_graph(all_edges)))))
    data = Data(x=vertices, edge_index=edges_transposed.t().contiguous())
    assert data.is_undirected()

    return data, starting_vertex


def generate_semirandom_hampath_graph(verts: int, delta_v: int, num_rand_edges: int, delta_e: int) -> tuple[Data, int]:
    v = verts + randint(-delta_v, delta_v)
    e = num_rand_edges + randint(-delta_e, delta_e)
    return generate_hampath_graph(v, e)


def generate_and_save_corpus(
        num_graphs: int, verts: int, delta_v: int, num_rand_edges: int, delta_e: int, out_path: str) -> None:
    """
    Generates random Hamiltonian-path-containing graphs with the specified number of vertices and random edges
    and saves them as serialized tensor files with filenames `graph0.pt`, ..., `graph{num_graphs}.pt` to out_path.

    @param num_graphs:
        the number of graphs to generate

    @param verts:
        the (roughly) mean number of vertices for each graph

    @param delta_v:
        the maximum allowed delta for random vertex count variation

    @param num_rand_edges:
        the (roughly) mean number of non-core-path edges for each graph

    @param delta_e:
        the maximum allowed delta for random edge count variation

    @param out_path:
        the directory in which to save the generated graphs
    """

    print('Generating graphs...')
    for i in tqdm(range(num_graphs)):
        data = generate_semirandom_hampath_graph(verts, delta_v, num_rand_edges, delta_e)
        torch.save(data, f'{out_path}/graph{i}.pt')


def load_corpus(path: str, load_count: int = -1) -> list[Data]:
    """
    Loads a corpus from directory path, optionally limiting to the first load_count entries.

    @param path:
        the directory containing the corpus

    @param load_count:
        the number of graphs to load, or -1 to load the entire corpus

    @return:
        a list of the deserialized graphs
    """
    graphs = []
    graph_num = 0
    next_graph: Callable[[], Path] = lambda: Path(path).joinpath(f'graph{graph_num}.pt')
    while load_count != 0 and next_graph().is_file():
        graphs.append(torch.load(next_graph()))
        graph_num += 1
        load_count -= 1

    if load_count > 0:
        print(f'Warning: only found {graph_num} out of {load_count} requested graphs')

    return graphs
