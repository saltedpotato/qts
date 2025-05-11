import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import KMeans
# AgglomerativeClustering, AffinityPropagation, DBSCAN


class Clustering:
    def __init__(self, df, reduction_threshold=10):
        self.df = df
        self.reduction_threshold = reduction_threshold
        self.stats = self.compute_stats()
        self.scaler = self.scaling()
        self.cluster_pairs = {}

    def compute_stats(self):
        return (
            self.df.select([i for i in self.df.columns if i not in ["date", "time"]])
            .with_columns(pl.all().pct_change())
            .mean()
            .transpose(include_header=True, header_name="ticker", column_names=["ret"])
            .join(
                self.df.select(
                    [i for i in self.df.columns if i not in ["date", "time"]]
                )
                .with_columns(pl.all().pct_change().std())
                .mean()
                .transpose(
                    include_header=True, header_name="ticker", column_names=["vol"]
                ),
                how="inner",
                left_on="ticker",
                right_on="ticker",
            )
        )

    def scaling(self):
        scaler = StandardScaler().fit_transform(self.stats.drop("ticker").to_numpy())

        return scaler

    def group_pairs(self, labels):
        for i in range(len(labels)):
            if labels[i] not in self.cluster_pairs.keys():
                self.cluster_pairs[labels[i]] = []

            self.cluster_pairs[labels[i]].append(self.stats["ticker"][i])

    def evaluate(self, silhouette_score):
        percentage_reduction = [0] * 2  # There is no reduction for the first two points

        for i in range(2, len(silhouette_score)):
            reduction = (
                (silhouette_score[i - 2] - silhouette_score[i])
                / silhouette_score[i - 2]
                * 100
            )
            percentage_reduction.append(reduction)

            if reduction < self.reduction_threshold:
                print(i-1)
                return i - 1

    def kmeans_clustering(self, min_clusters=5, max_clusters=20):
        silhouette_score = []

        for k in range(min_clusters, max_clusters):
            kmeans = KMeans(n_clusters=k, random_state=1111, n_init=10)
            kmeans.fit(self.scaler)
            silhouette_score.append(
                metrics.silhouette_score(self.scaler, kmeans.labels_, random_state=1111)
            )
        n = self.evaluate(silhouette_score=silhouette_score) + min_clusters
        kmeans = KMeans(n_clusters=n, random_state=1111, n_init=10)
        kmeans.fit(self.scaler)

        self.group_pairs(kmeans.labels_)
