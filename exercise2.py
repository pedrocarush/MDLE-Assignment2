import pyspark.sql.functions as F
from pyspark import Broadcast
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import ArrayType, DoubleType, IntegerType
from typing import Iterable, Any, Tuple
from argparse import ArgumentParser


try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Matplotlib could not be imported, graphs won\'t be plotted.')
    plt = None

from pyspark.ml.clustering import BisectingKMeans

def calculate_user_item_means(df: DataFrame,spark: SparkSession) -> Tuple[Broadcast, Broadcast, float]:
    user_means = {
        row['user_id']:row['user_avg']
        for row in df.groupBy('user_id')
            .agg(F.avg('rating').alias('user_avg'))
            .collect()
    }
    item_means = {
        row['item_id']:row['item_avg']
        for row in df.groupBy('item_id')
            .agg(F.avg('rating').alias('item_avg'))
            .collect()
    }
    mu = df.agg(F.avg('rating').alias('overall_avg')).collect()[0]['overall_avg']

    user_means = spark.sparkContext.broadcast(user_means)
    item_means = spark.sparkContext.broadcast(item_means)

    return user_means, item_means, mu

def cluster_with_bisectingkmeans(matrix: DataFrame, minDivisibleClusterSize: float=None, k: int=50, seed: int=1) -> DataFrame:

    if minDivisibleClusterSize is None:
        minDivisibleClusterSize = 1 / k

    bkm = BisectingKMeans(featuresCol='ratings', minDivisibleClusterSize=minDivisibleClusterSize, predictionCol="cluster_id").setK(k).setSeed(seed)
    model = bkm.fit(matrix)

    return model.transform(matrix)


def build_item_ratings_sparse_matrix(dataset: DataFrame,df:DataFrame) -> DataFrame:
    user_number = df.select('user_id').distinct().count()

    @F.udf(returnType=ArrayType(IntegerType(),False))
    def utility_matrix_row(elems: Iterable[Any]):

        sparse_row = [0] * user_number
        
        for (user_id,rating) in elems:
            sparse_row[user_id-1] = rating

        return sparse_row

    return (dataset.groupBy('item_id')
        .agg(F.collect_list(F.array('user_id','rating')).alias('ratings'))
        .withColumn('ratings', utility_matrix_row(F.col('ratings')).cast(ArrayType(DoubleType(),False)))
    )

def predict_ratings(test_set: DataFrame, train_matrix_with_clusters: DataFrame, user_means: Broadcast, item_means: Broadcast, mu: float) -> DataFrame:
    """Predict user ratings on an item using item-based Collaborative Filtering. Function built with batch processing in mind.

    The supplied utility matrix should have an item for each row and the following columns:
    - `item_id` containing the item ID
    - `ratings` containing the respective row of the utility matrix
    - `cluster_id` containing the ID of the cluster the item belongs to

    Parameters
    ----------
    test_set : DataFrame
        Spark dataframe containing the user and item IDs in columns `user_id` and `item_id` respectively.
        May contain other columns, but they should not overlap with the internal columns
    train_matrix_with_clusters : DataFrame
        Spark dataframe representing the utility matrix
    user_means : Broadcast
        Spark-broadcasted Python dictionary, containing the average rating for each user
    item_means : Broadcast
        Spark-broadcasted Python dictionary, containing the average rating for each item
    mu : float
        Average overall rating
    """

    return (test_set
        .join(train_matrix_with_clusters, on='item_id', how='inner') # get cluster and ratings of test item
        # the join should be done first so that we remove items and users that were not trained on (never seen before)
        .withColumn('item_mean', F.udf(lambda x: item_means.value[x], returnType=DoubleType())('item_id'))
        .withColumn('user_mean', F.udf(lambda x: user_means.value[x], returnType=DoubleType())('user_id'))
        .withColumn('ratings_pearson', F.transform('ratings', lambda x: x - F.col('item_mean')))
        .join(train_matrix_with_clusters
            .withColumnsRenamed({
                'item_id': 'other_item_id',
                'ratings': 'other_ratings',
            })
            .withColumn('other_item_mean', F.udf(lambda x: item_means.value[x], returnType=DoubleType())('other_item_id'))
            .withColumn('other_ratings_pearson', F.transform('other_ratings', lambda x: x - F.col('other_item_mean'))),
            on='cluster_id',
            how='inner')
        .withColumn('user_other_rating', F.col('other_ratings')[F.col('user_id') - 1])
        .filter(F.col('user_other_rating') != 0) # should remove the test item since it never gave got a rating from the test user
        .withColumn('similarity',
            F.aggregate(
                F.zip_with('ratings_pearson', 'other_ratings_pearson', lambda x1, x2: x1 * x2),
                initialValue=F.lit(0.0),
                merge=lambda acc, x: acc + x
            )
            /
            (
                F.sqrt(F.aggregate(
                    F.transform('ratings_pearson', lambda x: x**2),
                    initialValue=F.lit(0.0),
                    merge=lambda acc, x: acc + x
                ))
                *
                F.sqrt(F.aggregate(
                    F.transform('other_ratings_pearson', lambda x: x**2),
                    initialValue=F.lit(0.0),
                    merge=lambda acc, x: acc + x
                ))
            )
        )
        .withColumn('baseline', -mu + F.col('user_mean') + F.col('item_mean')) # b_xi
        .withColumn('other_baseline', -mu + F.col('user_mean') + F.col('other_item_mean')) # b_xj
        .groupBy(*test_set.columns, 'baseline')
        .agg((F.col('baseline') + F.sum(F.col('similarity') * (F.col('user_other_rating') - F.col('other_baseline'))) / F.sum('similarity')).alias('predicted_rating'))
        .drop('baseline')
    )

def main(
    dataset: str,
    seed_random: int,
    n_clusters: int        
):

    spark = SparkSession.builder.getOrCreate()

    df = (spark.read
        .option('header', 'false')
        .option('sep', '\t')
        .csv(dataset)
    )

    df = (df.withColumnRenamed('_c0', 'user_id') 
        .withColumnRenamed('_c1', 'item_id')
        .withColumnRenamed('_c2', 'rating')
        .drop('_c3')
        .select(F.col('user_id').cast(IntegerType()), F.col('item_id').cast(IntegerType()), F.col('rating').cast(IntegerType()))
    )

    train_set, test_set = df.randomSplit([9.0, 1.0], seed=seed_random)

    user_means, item_means, mu = calculate_user_item_means(train_set, spark)

    train_matrix = build_item_ratings_sparse_matrix(train_set,df)
    train_matrix_with_clusters = cluster_with_bisectingkmeans(train_matrix, minDivisibleClusterSize=1/n_clusters, k=n_clusters)

    predictions = predict_ratings(test_set.withColumnRenamed('rating', 'true_rating'), train_matrix_with_clusters, user_means, item_means, mu)

    non_tested = (test_set
        .join(train_matrix_with_clusters, on='item_id', how='left')
        .filter(F.col('cluster_id').isNull())
    ).count()

    tested_total = test_set.count()

    print(f'Portion of user-item pairs that were not tested: {non_tested/tested_total:%}')

    total_tested = test_set.join(train_matrix_with_clusters, on='item_id', how='inner').count()

    rmse = ((predictions
        .withColumn('square_error', (F.col('true_rating') - F.col('predicted_rating'))**2)
        .select('square_error')
        .groupBy()
        .sum('square_error')
        .collect()[0]['sum(square_error)']
        ) / (total_tested)) ** 0.5

    print('RMSE:', rmse)

if __name__ == '__main__':

    default_str = ' (default: %(default)s)'

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="path to the file housing the MovieLens dataset u.data" + default_str, default="./data/ml-100k/u.data")
    parser.add_argument("--seed", type=int, help="random seed" + default_str, default=1)
    parser.add_argument("--k", type=int, help="number of clusters" + default_str, default=50)
    parser.add_argument("--min-divisible-cluster-size", type=float, help="argument")

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        seed_random=args.seed,
        n_clusters=args.k
        
    )