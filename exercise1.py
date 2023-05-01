import json
import pickle
import os.path
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestCentroid
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, ArrayType, FloatType, DoubleType, IntegerType, StructField, StructType
from itertools import combinations
from typing import Iterable, Any, List, Sequence, Set, Tuple
from typing import Union
from argparse import ArgumentParser
from plot_helpers import heatmap, annotate_heatmap

"""
Python-file version of Exercise 1's Jupyter notebook, for submission via spark-submit.
The documentation is present in the notebook.
"""



@dataclass(init=False)
class SummarizedCluster:
    n:      int                     
    sum_:   npt.NDArray[np.float64]
    sumsq_: npt.NDArray[np.float64]
    id_:    Union[int, None]
    tracks: Set[int]

    def __init__(self, dimensions: int, id_: int=None):
        self.n = 0
        self.sum_ = np.zeros((dimensions,), dtype=np.float64)
        self.sumsq_ = np.zeros((dimensions,), dtype=np.float64)
        self.id_ = id_
        self.tracks = set()
    
    def summarize(self, point: npt.NDArray[np.float64], track_id: int):
        self.n += 1
        self.sum_ += point
        self.sumsq_ += point**2
        self.tracks.add(track_id)
    
    def summarize_points(self, points: npt.NDArray[np.float64], track_ids: Set[int]):
        self.n += points.shape[0]
        self.sum_ += np.sum(points, axis=0)
        self.sumsq_ += np.sum(points**2, axis=0)
        self.tracks |= track_ids

    def centroid(self) -> npt.NDArray[np.float64]:
        return self.sum_ / self.n

    def variance(self) -> npt.NDArray[np.float64]:
        return (self.sumsq_ / self.n) - (self.sum_ / self.n)**2

    def standard_deviation(self) -> npt.NDArray[np.float64]:
        return np.sqrt(self.variance())

    def __add__(self, other: 'SummarizedCluster') -> 'SummarizedCluster':
        if not isinstance(other, SummarizedCluster):
            raise ValueError(f"Addition is not supported between a SummarizedCluster and a '{type(other)}'.")
        if self.id_ is not None and other.id_ is not None and self.id_ != self.other:
            raise ValueError(f"Clusters {self} and {other} have different explicit ids ({self.id_} != {other.id_}).")
        if self.tracks & other.tracks:
            raise ValueError(f"The clusters {self} and {other} overlap each other.")
        res = SummarizedCluster(self.sum_.size, self.id_ if self.id_ is not None else other.id_)
        res.n = self.n + other.n
        res.sum_ = self.sum_ + other.sum_
        res.sumsq_ = self.sumsq_ + other.sumsq_
        res.tracks = self.tracks | other.tracks
        return res

    def __iadd__(self, other: 'SummarizedCluster') -> 'SummarizedCluster':
        if not isinstance(other, SummarizedCluster):
            raise ValueError(f"Addition is not supported between a SummarizedCluster and a '{type(other)}'.")
        if self.id_ is not None and other.id_ is not None and self.id_ != self.other:
            raise ValueError(f"Clusters {self} and {other} have different explicit ids ({self.id_} != {other.id_}).")
        if self.tracks & other.tracks:
            raise ValueError(f"The clusters {self} and {other} overlap each other.")
        self.id_ = self.id_ if self.id_ is not None else other.id_
        self.n = self.n + other.n
        self.sum_ = self.sum_ + other.sum_
        self.sumsq_ = self.sumsq_ + other.sumsq_
        self.tracks = self.tracks | other.tracks
        return self

    def __str__(self) -> str:
        return f'SummarizedCluster({self.id_}, n={self.n})'

    def __repr__(self) -> str:
        return str(self)
    

def read_tracks_df(spark: SparkSession, tracks_path: str) -> DataFrame:
    tracks_df = (spark.read
        .option("multiline", "true")
        .option("quote", '"')
        .option("escape", '"')
        .csv(tracks_path)
    )

    # rename columns with row values from first row to second row
    column_categories = list(zip(*tracks_df.take(2)))
    columns = tracks_df.columns
    tracks_df = tracks_df.select(F.col(columns[0]).alias('track_id'),
        *(F.col(column).alias("-".join(map(str, categories)))
        for column, categories in zip(columns[1:], column_categories[1:]))
    )

    tracks_df = (tracks_df
        .filter(F.col("track_id").isNotNull()) 
        .filter(F.col("track_id") != "track_id")
    )

    return tracks_df


def read_features_df(spark: SparkSession, features_path: str) -> DataFrame:
    features_df = (spark.read
        .csv(features_path)
    )

    # rename columns with row values from first row to second row
    column_categories = list(zip(*features_df.take(3)))
    columns = features_df.columns
    features_df = features_df.select(F.col(columns[0]).alias('track_id'),
        *(F.col(column).cast(DoubleType()).alias("-".join(map(str, categories)))
        for column, categories in zip(columns[1:], column_categories[1:]))
    )

    features_df = (features_df
        .filter(F.col("track_id") != "feature")
        .filter(F.col("track_id") != "statistics")
        .filter(F.col("track_id") != "number")
        .filter(F.col("track_id") != "track_id")
    )

    return features_df


def calculate_metrics(pd_df: pd.DataFrame, centroids: npt.ArrayLike) -> pd.DataFrame:
    """Calculate the metrics (radius, diameter, density_r, density_d) for each cluster"""

    cluster = pd_df["cluster"].values[0]
    metrics = pd.DataFrame({'radius': [0], 'diameter': [0],'density_r': [0],'density_d': [0]}, columns=['radius', 'diameter','density_r','density_d'])
    centroid = centroids[cluster].reshape(1,-1)

    matrix = pd_df.drop(columns=["cluster", "track_id"]).to_numpy()
    
    matrix_radius = np.sqrt(np.sum((matrix - centroid)**2, axis=1))
    metrics.loc[0,'radius'] = np.max(matrix_radius)
    # calculate density with radius
    metrics.loc[0,'density_r'] = len(pd_df) / metrics.loc[0,'radius']**2

    for i in range(matrix.shape[0]):
        matrix_diameter = np.sqrt(np.sum((matrix[i:,:] - matrix[i,:])**2, axis=1))

        max_diameter = np.max(matrix_diameter)
        if max_diameter > metrics.loc[0,'diameter']:
            metrics.loc[0,'diameter'] = max_diameter
        
    # calculate density with diameter
    metrics.loc[0,'density_d'] = len(pd_df) / metrics.loc[0,'diameter']**2

    return metrics


def cluster_agglomeratively(data: pd.DataFrame, n_clusters: int) -> npt.NDArray[np.float32]:
    if "cluster" not in data.columns:
        raise ValueError("The 'cluster' column should be present in the dataframe!")
    
    data_features_only = data.drop(columns=["cluster", "track_id"])

    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    clusterer.fit(data_features_only)
    
    centroid_calculator = NearestCentroid()
    centroid_calculator.fit(data_features_only, clusterer.labels_)

    data["cluster"] = clusterer.labels_
    return centroid_calculator.centroids_


def plot_standardized_heatmap(metrics_pd_array: List[pd.DataFrame], n_clusters_sequence: Sequence[int]):
    metrics_heatmap = np.array([
        [
            m["density_r"].mean(),
            m["density_d"].mean(),
            m["density_r"].var(),
            m["density_d"].var(),
        ]
        for m in metrics_pd_array
    ])

    metrics_heatmap_standardized = (metrics_heatmap - metrics_heatmap.mean(axis=0)) / metrics_heatmap.std(axis=0)

    fig, ax = plt.subplots()

    im, cbar = heatmap(
        data=metrics_heatmap_standardized.T,
        row_labels=["avg. density ($r^2$)", "avg. density ($d^2$)", "var. density ($r^2$)", "var. density ($d^2$)"],
        col_labels=list(map(str, n_clusters_sequence)),
        cbarlabel="Z-score",
        cbar_kw={"location": "bottom"},
        cmap="cool",
        ax=ax
    )
    
    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    ax.set_title("Standardized metrics for each of the cluster configurations")

    fig.tight_layout()
    fig.show()


def summarize_cluster_df(cluster_df: pd.DataFrame, discard_sets: List[SummarizedCluster]) -> None:
    cluster_id = cluster_df["cluster"].values[0]
    cluster_features_mtx = cluster_df.drop(columns=["cluster", "track_id"]).to_numpy()
    
    track_ids = set(cluster_df["track_id"].values)

    discard_set = discard_sets[cluster_id]
    discard_set.summarize_points(cluster_features_mtx, track_ids)


def mahalanobis_distance_pd(x: pd.DataFrame, s: SummarizedCluster) -> pd.Series:
    return (((x - s.centroid()) / s.standard_deviation())**2).sum(axis=1) ** 0.5


def plot_cluster_bars(
        spark: SparkSession, 
        tracks_df: DataFrame, 
        discard_sets: List[SummarizedCluster],
        dataset: str,
        result_path_number: str,
        result_path_portion: str):
    
    n_clusters = len(discard_sets)
    cluster_ids = [None] + list(range(n_clusters))
    
    cluster_assignment_df = spark.createDataFrame(
        data=[(track, ds.id_) for ds in discard_sets for track in ds.tracks],
        schema=StructType([StructField('track_id', StringType(), False), StructField('cluster_id', IntegerType(), False)])
    )

    genres_path = os.path.join(dataset, 'genres.csv')
    genres_colors_path = os.path.join(dataset, 'raw_genres.csv')

    genres_df = spark.read.option('header', 'true').csv(genres_path)
    genres_colors_df = spark.read.option('header', 'true').csv(genres_colors_path)

    genres = [row["title"] for row in genres_df.filter(F.col('parent') == 0).sort(F.col('genre_id').cast(IntegerType())).select('title').collect()]
    genres_colors = {row["title"]:row["genre_color"] for row in genres_df.filter(F.col('parent') == 0).join(genres_colors_df, on='genre_id', how='inner').collect()}

    to_list = F.udf(lambda s: json.loads(s), ArrayType(StringType(), False))

    # Total number analysis

    genre_counts_for_each_discard_set = {
        row['cluster_id']:row['genre_counts']
        for row in (tracks_df
            .select('track_id', F.explode(to_list('track-genres_all')).alias('genre_id'))
            .join(genres_df
                .filter(F.col('parent') == 0)
                .select('genre_id', F.col('title').alias('genre_title')),
                on='genre_id',
                how='inner')
            .drop('genre_id')
            .join(cluster_assignment_df, on='track_id', how='left')
            .select('track_id', 'cluster_id', 'genre_title')
            .groupby('cluster_id', 'genre_title')
            .agg(F.count('track_id').alias('count'))
            .groupby('cluster_id')
            .agg(F.map_from_arrays(F.collect_list('genre_title'), F.collect_list('count')).alias('genre_counts'))
        ).collect()
    }

    genre_counts = {
        attribute:np.array([(genre_counts_for_each_discard_set[cluster_id][attribute] if (cluster_id in genre_counts_for_each_discard_set and attribute in genre_counts_for_each_discard_set[cluster_id]) else 0) for cluster_id in cluster_ids])
        for attribute in genres
    }

    width = 0.75

    fig, ax = plt.subplots(figsize=(9, 7))
    bottom = np.zeros(n_clusters + 1) # include the outliers

    cluster_ids_xs = [-1] + list(range(n_clusters))

    for genre in genres:
        ax.bar(cluster_ids_xs, genre_counts[genre], width, label=genre, bottom=bottom, color=genres_colors[genre])
        bottom += genre_counts[genre]

    ax.set_title(f"Number of tracks per genre on each cluster")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_xticks(ticks=cluster_ids_xs, labels=map(str, cluster_ids))

    fig.savefig(result_path_number)

    # Top 3 analysis

    genre_counts_for_each_discard_set_top_3_portions = {
        discard_set:sorted(counts.items(), key=lambda t: t[1], reverse=True)[:3]
        for discard_set, counts in genre_counts_for_each_discard_set.items()
    }

    genre_counts_for_each_discard_set_top_3_portions = {
        discard_set:[(genre, count / sum(count for _, count in top_3)) for genre, count in top_3]
        for discard_set, top_3 in genre_counts_for_each_discard_set_top_3_portions.items()
    }

    genre_counts_top_3 = {}
    for discard_set, top_3 in genre_counts_for_each_discard_set_top_3_portions.items():
        for genre, portion in top_3:
            discard_portions = genre_counts_top_3.setdefault(genre, np.zeros((n_clusters + 1,)))
            if discard_set is None:
                discard_portions[0] = portion
            else:
                discard_portions[discard_set + 1] = portion
    
    fig, ax = plt.subplots(figsize=(9, 7))
    bottom = np.zeros(n_clusters + 1) # include the outliers

    cluster_ids_xs = [-1] + list(range(n_clusters))

    for genre, portions in genre_counts_top_3.items():
        p = ax.bar(cluster_ids_xs, portions, width, label=genre, bottom=bottom, color=genres_colors[genre])
        bottom += portions

    ax.set_title("Portion of tracks per genre in the top 3 genres of each cluster")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Portion")
    ax.set_xticks(ticks=cluster_ids_xs, labels=map(str, cluster_ids))

    fig.savefig(result_path_portion)



def assign_discard_sets(loaded_points_pd: pd.DataFrame, discard_sets: List[SummarizedCluster], cluster_distance_threshold: float) -> pd.DataFrame:
    n_clusters = len(discard_sets)
    prefix = "cluster_distance_"
    cluster_distance_columns = [f"{prefix}{i}" for i in range(n_clusters)]

    loaded_points_pd = pd.concat(
        objs=[loaded_points_pd, pd.DataFrame(
            data=np.zeros((loaded_points_pd.shape[0], len(cluster_distance_columns) + 2)),
            columns=cluster_distance_columns + ['min_cluster_distance', 'cluster_id'])],
        axis=1
    )

    for i, discard_set in enumerate(discard_sets):
        loaded_points_pd[f"cluster_distance_{i}"] = mahalanobis_distance_pd(loaded_points_pd.iloc[:, 1:519], discard_set)
    loaded_points_pd["min_cluster_distance"] = loaded_points_pd[cluster_distance_columns].min(axis=1)
    loaded_points_pd["cluster_id"] = loaded_points_pd[cluster_distance_columns].idxmin(axis=1).str.slice(start=len(prefix)).astype(np.int32)

    loaded_points_pd.drop(columns=cluster_distance_columns, inplace=True)

    # Don't consider the points that surpass the threshold for the discard sets
    loaded_points_pd.loc[loaded_points_pd["min_cluster_distance"] >= cluster_distance_threshold, "cluster_id"] = -1
    loaded_points_pd.drop(columns=["min_cluster_distance"], inplace=True)

    return loaded_points_pd


def print_progress(progress: float, message: str):
    print(f"[{progress:2%}] {message:100}", end="\r")


def main(
        dataset: str,
        small_metrics: bool,
        small_metrics_n_clusters_sequence: Sequence[int],
        small_metrics_results_path: str,
        bfr_n_clusters: int,
        bfr_max_memory_used_bytes: int,
        bfr_seed: int,
        bfr_cluster_distance_threshold_standard_deviations: float,
        bfr_compression_set_merge_variance_threshold: float,
        bfr_dbscan_eps: float,
        bfr_results_folder: str,
        bfr_include_compression_sets: bool,
):
    
    tracks_path = os.path.join(dataset, 'tracks.csv')
    features_path = os.path.join(dataset, 'features.csv')

    spark = SparkSession.builder.getOrCreate()
    
    tracks_df = read_tracks_df(spark, tracks_path)
    features_df = read_features_df(spark, features_path)

    small_tracks_df = tracks_df.filter(F.col("set-subset") == 'small')
    small_features_df = (features_df
        .join(small_tracks_df, "track_id", "left")
        .filter(F.col("set-subset").isNotNull())
        .select(features_df.columns)
    )

    small_features_pd = small_features_df.toPandas()
    small_features_pd = pd.concat(objs=[small_features_pd, pd.DataFrame(data=np.zeros((len(small_features_pd), 1), dtype=np.int32), columns=["cluster"])], axis=1)

    if small_metrics:
        metrics_pd_array = []

        if os.path.exists(small_metrics_results_path):
            with open(small_metrics_results_path, 'rb') as f:
                metrics_pd_array = pickle.load(f)

        else:
            for n_clusters in small_metrics_n_clusters_sequence:
                print(f"Clustering with {n_clusters} clusters")
                centroids = cluster_agglomeratively(small_features_pd, n_clusters)
                metrics_pd_array.append(small_features_pd.groupby("cluster").apply(calculate_metrics, centroids))
            
            with open(small_metrics_results_path, 'wb') as f:
                pickle.dump(metrics_pd_array, f)
        
        plot_standardized_heatmap(metrics_pd_array, small_metrics_n_clusters_sequence)

    else:
        features_music_columns = features_df.columns.copy()
        features_music_columns.remove('track_id')

        rows_per_iteration = bfr_max_memory_used_bytes // (8 * len(features_df.columns))
        cluster_distance_threshold = ((bfr_cluster_distance_threshold_standard_deviations**2) * len(features_music_columns)) ** 0.5

        centroids = cluster_agglomeratively(small_features_pd, bfr_n_clusters)

        discard_sets: List[SummarizedCluster] = [SummarizedCluster(len(features_music_columns), id_) for id_ in range(bfr_n_clusters)]
        compression_sets: List[SummarizedCluster] = []
        retained_set: pd.DataFrame = pd.DataFrame(data=[], columns=features_df.columns)

        small_features_pd.groupby("cluster").apply(summarize_cluster_df, discard_sets)

        features_without_small_df = (features_df
            .join(small_tracks_df, "track_id", "left")
            .filter(F.col("set-subset").isNull())
            .select(features_df.columns)
        )

        total_rows = features_without_small_df.count()
        split_weights = [1.0] * (1 + (total_rows // rows_per_iteration))

        split_dfs = features_without_small_df.randomSplit(split_weights, seed=bfr_seed)

        print_progress(0, "Initialized BFR")

        for split_idx, loaded_points_df in enumerate(split_dfs):
            progress = split_idx / len(split_weights)
            
            print_progress(progress, "Collecting split into memory...")
            loaded_points_pd = loaded_points_df.toPandas()

            print_progress(progress, "Clustering with the Mahalanobis distance...")  
            loaded_points_pd = assign_discard_sets(loaded_points_pd, discard_sets, cluster_distance_threshold)
            
            print_progress(progress, "Calculated and collected Mahalanobis distances")

            for cluster_id, cluster_pd in loaded_points_pd.groupby("cluster_id"):
                track_ids = cluster_pd["track_id"]
                features_list = cluster_pd[features_music_columns]

                print_progress(progress, f"Evaluating cluster {cluster_id}...")

                # Step 3 - check which points go to the discard sets
                if cluster_id != -1:
                    discard_set = discard_sets[cluster_id]
                    discard_set.summarize_points(features_list, set(track_ids))
                
                # Step 4 - check which points go to the compression sets or the retained set
                else:
                    cluster_with_retained_pd = pd.concat(objs=[cluster_pd, retained_set], axis=0)

                    # Use same distance as above
                    clusterer = DBSCAN(eps=bfr_dbscan_eps, metric='euclidean')
                    clusterer.fit(cluster_with_retained_pd[features_music_columns])

                    retained_set = cluster_with_retained_pd[clusterer.labels_ == -1]

                    mini_clusters = set(clusterer.labels_) - {-1}

                    # Create compression sets
                    compression_sets_temp = [SummarizedCluster(len(features_music_columns), None) for _ in mini_clusters]
                    for mini_cluster_id in mini_clusters:
                        compression_sets_temp[mini_cluster_id].summarize_points(
                            cluster_with_retained_pd[features_music_columns][clusterer.labels_ == mini_cluster_id], 
                            set(cluster_with_retained_pd['track_id'][clusterer.labels_ == mini_cluster_id])
                        )
                    
                    compression_sets.extend(compression_sets_temp)
                
            print_progress(progress, f"Finished evaluating clusters (compression sets: {len(compression_sets)}, retained set: {len(retained_set)})")

            # Step 5 - merge compression sets
            compressing = True
            while compressing:
                compressing = False
                merged_compression_sets = []
                compression_set_idxs_to_remove = set()

                for (idx_1, compression_set_1), (idx_2, compression_set_2) in combinations(enumerate(compression_sets), 2):
                    if idx_1 in compression_set_idxs_to_remove or idx_2 in compression_set_idxs_to_remove:
                        continue

                    merged_compression_set = compression_set_1 + compression_set_2
                    if merged_compression_set.variance().mean() < bfr_compression_set_merge_variance_threshold * max(compression_set_1.variance().mean(), compression_set_2.variance().mean()):
                        merged_compression_sets.append(merged_compression_set)
                        compression_set_idxs_to_remove.add(idx_1)
                        compression_set_idxs_to_remove.add(idx_2)
                        compressing = True
                
                compression_sets: List[SummarizedCluster] = [cs for i, cs in enumerate(compression_sets) if i not in compression_set_idxs_to_remove]
                compression_sets.extend(merged_compression_sets)

                print_progress(progress, f"Completed one compression (compression sets: {len(compression_sets)})")
            
            print_progress((split_idx + 1) / len(split_weights), f"Finished one of the splits (compression sets: {len(compression_sets)}, retained set: {len(retained_set)})")
            print()

        print('Number of elements in...')
        print('discard sets:', sum(ds.n for ds in discard_sets))
        print('compression sets:', sum(ds.n for ds in compression_sets))
        print('retained set:', retained_set.shape[0])

        # Step 6 - merge CS into DS, leave RS out for further analysis
        if bfr_include_compression_sets:
            compression_sets_closest_discard = [
                (cs, min(((ds.id_, np.sqrt(np.sum(np.square(cs.centroid() - ds.centroid())))) for ds in discard_sets), key=lambda t: t[1])[0])
                for cs in compression_sets
            ]

            for compression_set, discard_set_id in compression_sets_closest_discard:
                discard_sets[discard_set_id] = compression_set + discard_sets[discard_set_id]

        float_to_fname = lambda f: str(f).replace('.', '-')

        plot_cluster_bars(
            spark=spark,
            tracks_df=tracks_df,
            discard_sets=discard_sets,
            dataset=dataset,
            result_path_number=f'{bfr_results_folder}/clustering_number_'
                f'c{bfr_n_clusters}_'
                f'std{float_to_fname(bfr_cluster_distance_threshold_standard_deviations)}_'
                f'mvt{float_to_fname(bfr_compression_set_merge_variance_threshold)}_'
                f'eps{float_to_fname(bfr_dbscan_eps)}_'
                f'{"dc" if bfr_include_compression_sets else "d"}.png',
            result_path_portion=f'{bfr_results_folder}/clustering_portion_'
                f'c{bfr_n_clusters}_'
                f'std{float_to_fname(bfr_cluster_distance_threshold_standard_deviations)}_'
                f'mvt{float_to_fname(bfr_compression_set_merge_variance_threshold)}_'
                f'eps{float_to_fname(bfr_dbscan_eps)}_'
                f'{"dc" if bfr_include_compression_sets else "d"}.png',
        )



if __name__ == '__main__':

    default_str = ' (default: %(default)s)'

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="path to the folder housing the FMA dataset CSV files" + default_str, default="./data")
    parser.add_argument("--small-metrics", action='store_true', help="whether to perform test clustering on the small subset of the dataset only" + default_str)
    parser.add_argument("--sm-n-clusters-range", type=int, nargs=2, help="range of number of clusters to try for the small subset of the dataset" + default_str, default=(8, 17))
    parser.add_argument("--sm-results-path", type=str, help="path to the small metrics results file" + default_str, default="./results/metrics_pd_array_pickle.pkl")
    parser.add_argument("--bfr-n-clusters", type=int, help="number of clusters to use for the BFR algorithm" + default_str, default=9)
    parser.add_argument("--bfr-max-memory-used-bytes", type=int, help="maximum memory used by the BFR algorithm in bytes" + default_str, default=int(.1e9))
    parser.add_argument("--bfr-seed", type=int, help="seed to use for the BFR algorithm" + default_str, default=0)
    parser.add_argument("--bfr-cdt-std", type=float, help="cluster distance threshold in standard deviations to use for the BFR algorithm" + default_str, default=1)
    parser.add_argument("--bfr-cs-mvt", type=float, help="compression set merge variance threshold to use for the BFR algorithm" + default_str, default=1.001)
    parser.add_argument("--bfr-dbscan-eps", type=float, help="DBSCAN epsilon to use for the BFR algorithm" + default_str, default=1000)
    parser.add_argument("--bfr-results-folder", type=str, help="path of the folder in which the result BFR graph will be stored" + default_str, default="./results/graphs")
    parser.add_argument("--bfr-exclude-compression-sets", action='store_true', help="whether to include the compression sets into the final clusters" + default_str)

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        small_metrics=args.small_metrics,
        small_metrics_n_clusters_sequence=range(*args.sm_n_clusters_range),
        small_metrics_results_path=args.sm_results_path,
        bfr_n_clusters=args.bfr_n_clusters,
        bfr_max_memory_used_bytes=args.bfr_max_memory_used_bytes,
        bfr_seed=args.bfr_seed,
        bfr_cluster_distance_threshold_standard_deviations=args.bfr_cdt_std,
        bfr_compression_set_merge_variance_threshold=args.bfr_cs_mvt,
        bfr_dbscan_eps=args.bfr_dbscan_eps,
        bfr_results_folder=args.bfr_results_folder,
        bfr_include_compression_sets=not args.bfr_exclude_compression_sets
    )