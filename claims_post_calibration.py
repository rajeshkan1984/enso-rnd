import os
import traceback
from uuid import uuid4
from xpms_file_storage.file_handler import XpmsResource, LocalResource

import pandas as pd


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


def upload_to_minio(solution_id, doc_id, file_name, local_file_path):
    minio_file_path = f'{solution_id}/sol_outputs/{doc_id}/{file_name}.csv'
    lr = LocalResource(key=local_file_path)
    mr = XpmsResource().get(key=minio_file_path)
    lr.copy(mr)
    lr.delete()
    return mr.urn


def remove_local_file(local_file_path):
    if os.path.exists(local_file_path):
        os.remove(local_file_path)
    return


def filter_outliers(df, threshold, is_claim=False, top_n=5):

    if is_claim:
        # Group by claim_id and get the maximum score for outliers
        outlier_max_scores = (
            df[df['Final_Recommendation'] == 'Outlier']
            .groupby('SOURCE_CLAIM_ID')['Calibrated_Score']
            .max()
            .reset_index()
        )

        # Get claim_ids that pass threshold
        valid_claims = outlier_max_scores[
            outlier_max_scores['Calibrated_Score'] >= threshold
            ]

        # Get top N claims from valid claims based on score
        top_n_claims = (
            valid_claims
            .sort_values('Calibrated_Score', ascending=False)
            .head(top_n)
        )

        # Get all rows for top N claims including service line if any
        top_n_df = df[df['SOURCE_CLAIM_ID'].isin(top_n_claims['SOURCE_CLAIM_ID'])]

    else:
        # Filter outliers that meet threshold
        valid_rows = df[
            (df['Final_Recommendation'] == 'Outlier') & (df['Calibrated_Score'] >= threshold)
            ]

        top_n_df = valid_rows.nlargest(top_n, 'Calibrated_Score')

    return top_n_df


def claims_post_calibration(config=None, **objects):
    try:
        file_path = objects["output_path"] if "output_path" in objects.keys() else \
            objects["document"][0]["metadata"]["properties"]["file_metadata"]["file_path"]
        no_of_claims = config.get('no_of_claims', 10)
        is_claim = config.get('is_claim', False)
        threshold = config.get('threshold', 60)

        local_file_path = download_minio_file(file_path)
        df = read_file(local_file_path)

        df_outlier = df[df['Final_Recommendation'] == 'Outlier']
        local_path = "/tmp/{}.csv".format(str(uuid4()))
        df_outlier.to_csv(local_path, index=False)
        minio_path = upload_to_minio(config["context"]["solution_id"], objects['doc_id'], 'outlier', local_path)
        output_json = {"data_format": "csv", "output_path": minio_path, "root_id": objects['doc_id'],
                      "doc_id": objects['doc_id']}
        filter_df = filter_outliers(df, threshold, is_claim, no_of_claims)
        local_path = "/tmp/{}.csv".format(str(uuid4()))
        filter_df.to_csv(local_path, index=False)
        minio_path = upload_to_minio(config["context"]["solution_id"], objects['doc_id'], 'filtered_post_calibration',
                                     local_path)
        return output_json

    except Exception as e:
        return traceback.format_exc()
