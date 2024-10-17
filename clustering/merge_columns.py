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


def merge_columns(config=None, **objects):
    cols_datatype = config.get("cols_datatype",None)
    separator = config.get("separator",None)
    impute_value = config.get("impute_value", "NA")
    delete_merged_cols = config.get("delete_merged_cols", False)
    column_names = config.get("column_names", None)
    new_column_name = config.get("new_column_name", None)

    file_path = objects['document'][0]['metadata']['properties']['file_metadata']['file_path']
    doc_id = objects['document'][0]['doc_id']
    solution_id = objects['document'][0]['solution_id']
    local_file_path = Util.download_minio_file(file_path)
    df = Util.read_file(local_file_path)

    if isinstance(column_names,str):
        column_names = column_names.split(",")
    missing_columns = set(column_names) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Column(s) {missing_columns} not found in the DataFrame.")

    cols_datatype = [cols_datatype] if cols_datatype else [str(dtype) for dtype in
                                                                               df[column_names].dtypes]
    temp_df = df.copy()
    if any(dtype in ["string", "object"] for dtype in cols_datatype):
        separator = separator if separator else ''
        temp_df = temp_df.astype({k: str for k in column_names})
        df[new_column_name] = temp_df[column_names].apply(
            lambda x: separator.join(sorted(filter(lambda val: not pd.isna(val) and str(val) != "nan", x))),
            axis=1
        )
    elif all(dtype in ["numeric", "int64", "float64"] for dtype in cols_datatype):
        temp_df[column_names] = temp_df[column_names].apply(pd.to_numeric, errors='coerce')
        df[new_column_name] = temp_df[column_names].sum(axis=1)

    if impute_value is not None:
        df[new_column_name] = df[new_column_name].replace("", np.nan).fillna(impute_value)
    if delete_merged_cols == "True":
        df.drop(columns=column_names, inplace=True)

    local_file_path = f"/tmp/{doc_id}.csv"
    df.to_csv(local_file_path, index=False)
    minio_path = Util.upload_to_minio(solution_id, local_file_path,minio_path=file_path)
    return objects