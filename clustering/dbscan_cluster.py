import os
import pandas as pd
import numpy as np
import traceback

from xpms_file_storage.file_handler import XpmsResource, LocalResource

from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


class Util:
    @staticmethod
    def download_minio_file(file_path):
        ext = os.path.splitext(file_path)[-1]
        local_file_path = "/tmp/" + file_path.split('/')[-1]
        xrm = XpmsResource()
        mr = xrm.get(urn=file_path)
        lr = LocalResource(key=local_file_path)
        mr.copy(lr)
        return local_file_path

    @staticmethod
    def read_file(file_path):
        extn = file_path.split(".")[-1]
        if extn.lower() in ['xls', 'xlsx']:
            df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
        elif extn.lower() in ['csv']:
            df = pd.read_csv(file_path)
        return df

    @staticmethod
    def upload_to_minio(solution_id, local_file_path):
        filename = local_file_path.split('/')[-1]
        minio_file_path = f'{solution_id}/clustering/{filename}'
        lr = LocalResource(key=local_file_path)
        mr = XpmsResource().get(key=minio_file_path)
        lr.copy(mr)
        lr.delete()
        return minio_file_path


class Dbscan:
    n_neighbors = None
    eps = None
    min_samples = None
    min_sample_selection = None

    def __init__(self, config=None, df=None):
        self.__dict__.update(**config)
        self.config = config
        self.df = df

    def calculate_eps(self, n_neighbors):
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(self.df)
        distances, _ = neigh.kneighbors(self.df)
        return np.mean(distances[:, -1])

    def get_min_sample(self):
        if self.min_sample_selection == "Dimension + 1":
            return int(self.df.shape[1] + 1)
        elif self.min_sample_selection == "Dimension * 1.25":
            return int(self.df.shape[1] * 1.25)
        if self.min_sample_selection == "Dimension * 1.5":
            return int(self.df.shape[1] * 1.5)
        if self.min_sample_selection == "Dimension * 2":
            return int(self.df.shape[1] * 2)

    def run(self):
        if not self.eps:
            self.eps = self.calculate_eps(n_neighbors=self.n_neighbors if self.n_neighbors else self.get_min_sample())
        if not self.min_samples:
            self.min_samples = self.get_min_sample()
        clustering = DBSCAN(eps=self.eps + 1, min_samples=self.min_samples).fit(self.df)
        labels = clustering.labels_
        return labels


def calculate_silhouette(df, labels):
    if len(set(labels)) > 2 and -1 in labels:
        valid_data = df[df['clustering_prediction'] != -1].drop(columns=['clustering_prediction'])
        valid_labels = labels[labels != -1]

        silhouette_avg = silhouette_score(valid_data, valid_labels)

    else:
        silhouette_avg = None

    return silhouette_avg


def dbscan_cluster(config=None, **objects):
    try:
        file_path = objects['document'][0]['metadata']['properties']['file_metadata']['file_path']
        doc_id = objects['document'][0]['doc_id']
        solution_id = objects['document'][0]['solution_id']
        local_file_path = Util.download_minio_file(file_path)
        df = Util.read_file(local_file_path)
        identifier = config.get("identifier_columns")
        identifiers = df[identifier]
        df = df.drop(columns=[identifier])
        dbscan = Dbscan(config=config, df=df)
        labels = dbscan.run()
        df['clustering_prediction'] = labels

        silhouette_avg = calculate_silhouette(df, labels)

        df[identifier] = identifiers
        cluster_dict = {}
        for cluster_label in set(labels):
            claim_ids = df[df['clustering_prediction'] == cluster_label][identifier].tolist()
            cluster_dict[str(cluster_label)] = claim_ids

        local_file_path = f"/tmp/{doc_id}.csv"
        df.to_csv(local_file_path, index=False)
        minio_path = Util.upload_to_minio(solution_id, local_file_path)
        labels = pd.Series(labels).value_counts().to_dict()

        return {"output_path": minio_path,
                "claim_recommendation": {
                    "silhouette_score":float(silhouette_avg) if silhouette_avg else "not able to calculate",
                    "cluster_labels": cluster_dict
                }}
    except Exception as e:
        return traceback.format_exc()
