import os
import traceback
from uuid import uuid4
from xpms_file_storage.file_handler import XpmsResource, LocalResource

import pandas as pd

drop_columns = ['Semisupervised_Confidence', 'Semisupervised_Recommendation', 'clustering_prediction',
                'scaled_Confidence Score', 'scaled_Semisupervised_Confidence']


def download_minio_file(file_path):
    ext = os.path.splitext(file_path)[-1]
    local_file_path = "/tmp/{}{}".format(str(uuid4()), ext)
    xrm = XpmsResource()
    mr = xrm.get(urn=file_path)
    lr = LocalResource(key=local_file_path)
    mr.copy(lr)
    return local_file_path


def read_file(file_path):
    df = pd.read_csv(file_path)
    remove_local_file(file_path)
    return df


def upload_to_minio(solution_id, doc_id, local_file_path):
    minio_file_path = f'{solution_id}/sol_outputs/{doc_id}/filtered_calibration.csv'
    lr = LocalResource(key=local_file_path)
    mr = XpmsResource().get(key=minio_file_path)
    lr.copy(mr)
    lr.delete()
    return mr.urn


def remove_local_file(local_file_path):
    if os.path.exists(local_file_path):
        os.remove(local_file_path)
    return



def filter_column(config=None, **objects):
    try:
        file_path = objects["output_path"] if "output_path" in objects.keys() else \
            objects["document"][0]["metadata"]["properties"]["file_metadata"]["file_path"]
        local_file_path = download_minio_file(file_path)
        df = read_file(local_file_path)
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
        local_path = "/tmp/{}.csv".format(str(uuid4()))
        df.to_csv(local_path, index=False)
        minio_path = upload_to_minio(config["context"]["solution_id"], objects['doc_id'], local_path)
        output_json = {"data_format": "csv", "output_path": minio_path, "root_id": objects['doc_id'],
                       "doc_id": objects['doc_id']}
        return output_json

    except Exception as e:
        return traceback.format_exc()
