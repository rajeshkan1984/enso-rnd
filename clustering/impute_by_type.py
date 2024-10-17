import os
import pandas as pd
import numpy as np
import traceback

from xpms_file_storage.file_handler import XpmsResource, LocalResource


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
    def upload_to_minio(solution_id, local_file_path,minio_path= None):
        filename = local_file_path.split('/')[-1]
        minio_file_path = minio_path if minio_path else f'{solution_id}/clustering/{filename}'
        lr = LocalResource(key=local_file_path)
        mr = XpmsResource().get(urn=minio_file_path) if "minio" in minio_file_path else XpmsResource().get(key=minio_file_path)
        lr.copy(mr)
        lr.delete()
        return minio_file_path


def impute_by_type(config=None, **objects):
    columns = config.get("columns",None)
    impute_value = config.get("impute_value", "mean")


    file_path = objects['document'][0]['metadata']['properties']['file_metadata']['file_path']
    doc_id = objects['document'][0]['doc_id']
    solution_id = objects['document'][0]['solution_id']
    local_file_path = Util.download_minio_file(file_path)
    df = Util.read_file(local_file_path)

    if columns:
        column_names = columns.split(",")
        dict_col = {col: impute_value for col in column_names}
        cols_with_type = dict_col
        type = {}
        if not cols_with_type:
            raise ValueError("The parameter cannot be empty. Please provide a valid Input.")
        for k, v in cols_with_type.items():
            v = v.lower()
            if k not in df.columns:
                raise ValueError(f"Feature '{k}' not found.")
            if v == "mean":
                mean = df[k].mean()
                df[k] = df[k].fillna(mean)
                type[v] = mean
            elif v == "median":
                median = df[k].median()
                df[k] = df[k].fillna(median)
                type[v] = median
            elif v == "mode":
                mode = df[k].mode()[0]
                df[k] = df[k].fillna(mode)
                type[v] = mode
            else:
                raise ValueError(
                    f"Invalid Imputation type '{v}' . Valid types are 'mean', 'median', or 'mode'.")

    local_file_path = f"/tmp/{doc_id}.csv"
    df.to_csv(local_file_path, index=False)
    minio_path = Util.upload_to_minio(solution_id, local_file_path,minio_path=file_path)
    return objects