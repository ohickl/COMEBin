import os
import sys
import time
import logging

import numpy as np
import pandas as pd

from typing import List, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import hnswlib
import leidenalg
import functools

import scipy.sparse as sp

from igraph import Graph
from sklearn.preprocessing import normalize
from sklearn.cluster._kmeans import euclidean_distances, stable_cumsum, KMeans, check_random_state, row_norms, MiniBatchKMeans

from utils import get_length, calculateN50, save_result
from scripts.gen_bins_from_tsv import gen_bins as gen_bins_from_tsv

logger = logging.getLogger('COMEBin')

logger.setLevel(logging.INFO)

# logging
formatter = logging.Formatter('%(asctime)s - %(message)s')

console_hdr = logging.StreamHandler()
console_hdr.setFormatter(formatter)

logger.addHandler(console_hdr)


def fit_hnsw_index(logger, features,num_threads, ef: int = 100, M: int = 16,
                   space: str = 'l2', save_index_file: bool = False) -> hnswlib.Index:
    """
    Fit an HNSW index with the given features using the HNSWlib library; Convenience function to create HNSW graph.

    :param logger: The logger object for logging messages.
    :param features: A list of lists containing the embeddings.
    :param ef: The ef parameter to tune the HNSW algorithm (default: 100).
    :param M: The M parameter to tune the HNSW algorithm (default: 16).
    :param space: The space in which the index operates (default: 'l2').
    :param save_index_file: The path to save the HNSW index file (optional).

    :return: The HNSW index created using the given features.

    This function fits an HNSW index to the provided features, allowing efficient similarity search in high-dimensional spaces.
    """

    time_start = time.time()
    num_elements = len(features)
    labels_index = np.arange(num_elements)
    EMBEDDING_SIZE = len(features[0])

    # Declaring index
    # possible space options are l2, cosine or ip
    p = hnswlib.Index(space=space, dim=EMBEDDING_SIZE)

    # Initing index - the maximum number of elements should be known
    p.init_index(max_elements=num_elements, ef_construction=ef, M=M)

    # Element insertion
    int_labels = p.add_items(features, labels_index, num_threads=num_threads)

    # Controlling the recall by setting ef
    # ef should always be > k
    p.set_ef(ef)

    # If you want to save the graph to a file
    if save_index_file:
        p.save_index(save_index_file)
    time_end = time.time()
    logger.info('Time cost:\t' +str(time_end - time_start) + "s")
    return p


def seed_kmeans_full(logger, contig_file: str, namelist: List[str], out_path: str,
                     X_mat: np.ndarray, bin_number: int, prefix: str, length_weight: np.ndarray, seed_bacar_marker_url: str):
    """
    Perform weighted seed-kmeans clustering with specified parameters.

    Parameters:
    :param contig_file: The path to the contig file.
    :param namelist: A list of contig names.
    :param out_path: The output path for saving results.
    :param X_mat: The input data matrix for clustering.
    :param bin_number: The number of bins (clusters) to create.
    :param prefix: A prefix to be added to the output file names.
    :param length_weight: The weights for contig lengths.
    :param seed_bacar_marker_url: The path to the seed markers used for initialization.

    :return: None

    This function performs weighted seed-based k-means clustering on the input data using specified parameters and saves the results.
    """
    # outpath is a Path object and prefix is a string
    out_path = str(out_path) + prefix
    seed_bacar_marker_idx = gen_seed_idx(seed_bacar_marker_url, contig_id_list=namelist)
    time_start = time.time()
    # run seed-kmeans; length weight
    output_temp = out_path + '_k_' + str(
        bin_number) + '_result.tsv'
    if not (os.path.exists(output_temp)):
        km = KMeans(n_clusters=bin_number, n_jobs=-1, random_state=7, algorithm="full",
                    init=functools.partial(partial_seed_init, seed_idx=seed_bacar_marker_idx))
        km.fit(X_mat, sample_weight=length_weight)
        idx = km.labels_
        save_result(idx, output_temp, namelist)

        gen_bins_from_tsv(contig_file, output_temp, output_temp+'_bins')

        time_end = time.time()
        logger.info("Running weighted seed-kmeans cost:\t"+str(time_end - time_start) + 's.')


def gen_seed_idx(seedURL: str, contig_id_list: List[str]) -> List[int]:
    """
    Generate a list of indices corresponding to seed contig IDs from a given URL.

    :param seedURL: The URL or path to the file containing seed contig names.
    :param contig_id_list: List of all contig IDs to match with the seed contig names.
    :return: List[int]
    """
    seed_list = []
    with open(seedURL) as f:
        for line in f:
            if line.rstrip('\n') in contig_id_list:
                seed_list.append(line.rstrip('\n'))
    name_map = dict(zip(contig_id_list, range(len(contig_id_list))))
    seed_idx = [name_map[seed_name] for seed_name in seed_list]
    return seed_idx


# change from sklearn.cluster.kmeans
def partial_seed_init(X, n_clusters: int, random_state, seed_idx, n_local_trials=None) -> np.ndarray:
    """
    Partial initialization of KMeans centers with seeds from seed_idx.

    Parameters:
    :param X: Features.
    :param n_clusters: The number of clusters.
    :param random_state: Determines random number generation for centroid initialization. Use an int for reproducibility.
    :param seed_idx: Indices of seed points for initialization.
    :param n_local_trials: The number of local seeding trials. Default is None.

    Returns:
    :return centers (ndarray): The initialized cluster centers.

    This function initializes a KMeans clustering by partially seeding the centers with provided seeds.
    It is a modification of the KMeans initialization algorithm.
    """
    random_state = check_random_state(random_state)
    x_squared_norms = row_norms(X, squared=True)

    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly

    center_id = seed_idx[0]

    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)

    for c, center_id in enumerate(seed_idx[1:], 1):
        if sp.issparse(X):
            centers[c] = X[center_id].toarray()
        else:
            centers[c] = X[center_id]
        closest_dist_sq = np.minimum(closest_dist_sq,
                                     euclidean_distances(
                                         centers[c, np.newaxis], X, Y_norm_squared=x_squared_norms,
                                         squared=True))
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(len(seed_idx), n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


def run_leiden(output_file: str, namelist: List[str],
               ann_neighbor_indices: np.ndarray, ann_distances: np.ndarray,
               length_weight: List[float], max_edges: int, norm_embeddings: np.ndarray,
               bandwidth: float = 0.1, lmode: str = 'l2', initial_list: Optional[List[Union[int, None]]] = None,
               is_membership_fixed: Optional[List[bool]] = None, resolution_parameter: float = 1.0,
               partgraph_ratio: int = 50) -> None:
    try:
        # Input validation
        if lmode not in ['l1', 'l2']:
            raise ValueError("lmode must be either 'l1' or 'l2'")
        
        if not (0 <= partgraph_ratio <= 100):
            raise ValueError("partgraph_ratio must be between 0 and 100")

        # Prepare edge data
        sources = np.arange(len(norm_embeddings))[:, np.newaxis].repeat(max_edges, axis=1)
        targets = ann_neighbor_indices[:, 1:]
        weights = ann_distances[:, 1:]

        # Flatten and filter edges
        sources = sources.flatten()
        targets = targets.flatten()
        weights = weights.flatten()

        dist_cutoff = np.percentile(weights, partgraph_ratio)
        mask = weights <= dist_cutoff
        sources = sources[mask]
        targets = targets[mask]
        weights = weights[mask]

        # Apply distance transformation
        if lmode == 'l1':
            weights = np.exp(-np.sqrt(weights) / bandwidth)
        else:  # l2
            weights = np.exp(-weights / bandwidth)

        # Create undirected graph (remove duplicate edges)
        mask = sources < targets
        sources, targets, weights = sources[mask], targets[mask], weights[mask]

        # Create graph
        g = Graph(n=len(norm_embeddings), edges=list(zip(sources, targets)), directed=False)
        g.es['weight'] = weights

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBERVertexPartition,
            weights='weight',
            node_sizes=length_weight,
            initial_membership=initial_list,
            resolution_parameter=resolution_parameter,
            n_iterations=-1
        )

        # If is_membership_fixed is provided, optimize the partition
        if is_membership_fixed is not None:
            optimiser = leidenalg.Optimiser()
            optimiser.optimise_partition(partition, is_membership_fixed=is_membership_fixed, n_iterations=-1)

        # Prepare results
        contig_labels_dict = {name: f'group{community}' for name, community in zip(namelist, partition.membership)}

        # Write results to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w') as f:
            for contig in namelist:
                f.write(f"{contig}\t{contig_labels_dict[contig]}\n")

        logger.info(f"Results written to {output_file}")

    except Exception as e:
        logger.error(f"Error in run_leiden: {str(e)}")
        raise

    return None

def cluster(logger, args, prefix: Optional[str] = None) -> None:
    """
    Cluster contigs and save the results.

    :param args: Command-line arguments and settings.
    :param prefix: A prefix for the clustering mode (optional).
    :return: None
    """
    logger.info("Start clustering.")

    # Load and preprocess data
    emb_file = Path(args.emb_file)
    seed_file = Path(args.seed_file)
    output_path = Path(args.output_path) / 'cluster_res'
    contig_file = Path(args.contig_file)
    contig_len = args.contig_len

    output_path.mkdir(parents=True, exist_ok=True)

    # Load embeddings and contig names
    embMat, namelist = load_embeddings(emb_file)
    
    # Get contig lengths and filter short contigs
    lengths = get_length(contig_file)
    length_weight = [lengths[seq_id] for seq_id in namelist]
    
    mask = np.array(length_weight) >= contig_len
    embMat = embMat[mask]
    namelist = namelist[mask]
    length_weight = list(np.array(length_weight)[mask])

    N50 = calculateN50(length_weight)
    logger.info(f'N50:\t{N50}')

    # Normalize embeddings if required
    norm_embeddings = embMat if args.not_l2normaize else normalize(embMat)

    # Run weighted seed k-means
    run_weighted_seed_kmeans(args, contig_file, namelist, output_path, norm_embeddings, length_weight, seed_file, prefix)

    # Run Leiden clustering
    run_leiden_clustering(args, norm_embeddings, namelist, length_weight, seed_file, output_path)

    logger.info('Clustering completed.')

def load_embeddings(emb_file: Path) -> tuple:
    embHeader = pd.read_csv(emb_file, sep='\t', nrows=1)
    embMat = pd.read_csv(emb_file, sep='\t', usecols=range(1, embHeader.shape[1])).values
    namelist = pd.read_csv(emb_file, sep='\t', usecols=[0]).values.flatten()
    return embMat, namelist

def run_weighted_seed_kmeans(args, contig_file, namelist, output_path, norm_embeddings, length_weight, seed_file, prefix):
    try:
        seed_namelist = pd.read_csv(seed_file, header=None, sep='\t', usecols=[0]).values.flatten()
        seed_num = len(np.unique(seed_namelist))
    except pd.errors.EmptyDataError:
        logger.warning("Seed file is empty. Exiting now.")
        DB_EMPTY_ERROR = 66

        # Touch empty `comebin_res_bins` dir and exit (output_path is .../comebin_res/cluster_res)
        bin_dir = output_path.parent / 'comebin_res_bins'
        bin_dir.mkdir(parents=True, exist_ok=True) 

        sys.exit(DB_EMPTY_ERROR)

    mode = f'weight_seed_kmeans{"_" + prefix if prefix else ""}'
    logger.info("Run weighted seed k-means for obtaining the SCG information of the contigs within a manageable time during the final step.")
    
    bin_nums = [seed_num]
    if args.cluster_num:
        bin_nums.append(args.cluster_num)

    logger.info(f"Bin_numbers:\t{bin_nums}")
    for k in bin_nums:
        logger.info(k)
        seed_kmeans_full(logger, contig_file, namelist, output_path, norm_embeddings, k, mode, length_weight, seed_file)

def run_leiden_clustering(args, norm_embeddings, namelist, length_weight, seed_file, output_path):
    num_workers = args.num_threads
    parameter_list = [1, 5, 10, 30, 50, 70, 90, 110]
    bandwidth_list = [0.05, 0.1, 0.15, 0.2, 0.3]
    partgraph_ratio_list = [50, 80, 100]
    max_edges = 100

    p = fit_hnsw_index(logger, norm_embeddings, num_workers, ef=max_edges * 10)
    seed_bacar_marker_idx = gen_seed_idx(seed_file, contig_id_list=namelist)
    initial_list = list(range(len(namelist)))
    is_membership_fixed = [i in seed_bacar_marker_idx for i in initial_list]

    time_start = time.time()
    ann_neighbor_indices, ann_distances = p.knn_query(norm_embeddings, max_edges + 1, num_threads=num_workers)
    time_end = time.time()
    logger.info(f'knn query time cost:\t{time_end - time_start}s')

    total_tasks = sum(1 for _ in partgraph_ratio_list for _ in bandwidth_list for _ in parameter_list)

    # Limit Leidne worker count to not go OOM or get into IO bottleneck
    leiden_workers = int(max(1, num_workers // 1.5))

    logger.info(f'Start Leiden clustering with {leiden_workers} workers to process {total_tasks} tasks.')

    with ProcessPoolExecutor(max_workers=leiden_workers) as executor:
        futures = []
        for partgraph_ratio in partgraph_ratio_list:
            for bandwidth in bandwidth_list:
                for para in parameter_list:
                    output_file = output_path / f'Leiden_bandwidth_{bandwidth}_res_maxedges{max_edges}respara_{para}_partgraph_ratio_{partgraph_ratio}.tsv'
                    if not output_file.exists():
                        futures.append(executor.submit(
                            run_leiden, str(output_file), namelist, ann_neighbor_indices, ann_distances, 
                            length_weight, max_edges, norm_embeddings, bandwidth, 'l2', 
                            initial_list, is_membership_fixed, para, partgraph_ratio
                        ))

        with tqdm(total=total_tasks, desc="Leiden clustering") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in Leiden clustering: {str(e)}")
                pbar.update(1)

    logger.info('Leiden clustering completed')
