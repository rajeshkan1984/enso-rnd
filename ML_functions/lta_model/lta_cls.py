import json
import os
import shap
import pandas as pd
import numpy as np
from uuid import uuid4
from typing import Dict
from xpms_helper.model import model_utils
from xpms_storage.db_handler import DBProvider
from xpms_helper.executions.execution_variables import ExecutionVariables
from xpms_helper.model.data_schema import DatasetFormat, DatasetConvertor
from xpms_file_storage.file_handler import XpmsResourceFactory, LocalResource, XpmsResource


def get_doc_id(solution_id, dag_execution_id):
    obj = DBProvider.get_instance(db_name=solution_id)
    filter_obj = {"execution_id": dag_execution_id}
    res = obj.find(table="dag_task_executions", filter_obj=filter_obj, multi_select=False)
    inputs = json.loads(res['inputs'][0])
    if 'document' in inputs:
        doc_id = inputs['document'][0]['doc_id']
    elif 'data' in inputs and 'doc_id' in inputs['data']:
        doc_id = inputs['data']['doc_id']
    else:
        doc_id = None
    return doc_id


def get_shap_values(model, X_processed: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification, take class 1
    return pd.DataFrame(shap_values, columns=X_processed.columns, index=X_processed.index)


def create_shap_dict(shap_row: pd.Series) -> Dict[str, float]:
    # Convert series to dictionary and sort by absolute value
    shap_dict = dict(shap_row)
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    return {k: round(float(v), 6) for k, v in sorted_shap}


def upload_to_minio(solution_id, local_file_path, doc_id="train"):
    minio_file_path = f'{solution_id}/sol_outputs/{doc_id}/lta_recommendations.csv'
    lr = LocalResource(key=local_file_path)
    mr = XpmsResource().get(key=minio_file_path)
    lr.copy(mr)
    lr.delete()
    return minio_file_path


def train(datasets, config):
    dataset = DatasetConvertor.convert(datasets, DatasetFormat.DATA_FRAME, None)
    train_info = {"name": "SVC"}
    return train_info, dataset


# %%
def run(datasets, config):
    solution_id = config["context"]["solution_id"]
    dag_execution_id = config["context"]["dag_execution_id"]
    doc_id = get_doc_id(solution_id, dag_execution_id) if dag_execution_id else config["context"]["doc_id"]
    dataset = DatasetConvertor.convert(datasets, DatasetFormat.DATA_FRAME, None)
    run_df = dataset["value"]
    target_column = "adjustment_status"
    # run_df = run_df.loc[:, run_df.columns != target_column]
    X = run_df.loc[:, run_df.columns != target_column]
    file_name = "lta_cls_model.pkl"
    model_obj = model_utils.load(file_name=file_name, config=config)
    predictions = model_obj.predict_proba(X)
    result_df = pd.DataFrame(data=predictions, columns=model_obj.classes_)
    y_predictions = (predictions[:, 1] >= 0.5).astype(int)
    result_dataset = {"value": result_df, "data_format": "data_frame"}
    shap_values_df = get_shap_values(model_obj, X)
    shap_values_df = shap_values_df.apply(create_shap_dict, axis=1)
    claim_ids = ExecutionVariables.get_variable(config['context']['solution_id'], 'claim_ids')
    recommendations_df = pd.DataFrame({
        'clm_id': claim_ids,
        'predicted_class': y_predictions,
        'recommendation': np.where(y_predictions == 1, 'Adjusted', 'Valid'),
        'confidence_score': predictions[:, 1]  # Confidence in prediction
    })
    recommendations_df['shap_values'] = shap_values_df
    local_file_path = f"/tmp/{str(uuid4())}.csv"
    recommendations_df.to_csv(local_file_path, index=False)
    doc_id = doc_id if doc_id else "train"
    minio_path = upload_to_minio(solution_id, local_file_path, doc_id)
    return result_dataset


# %%
def evaluate(datasets, config):
    dataset = DatasetConvertor.convert(datasets, DatasetFormat.DATA_FRAME, None)
    if "scorers" in config:
        scorers = config["scorers"]
    else:
        scorers = ["accuracy"]
    eval_df = dataset["value"]
    target_colum = "class"
    y = eval_df[target_colum]
    model_output = run(datasets, config)
    y_pred = model_output["value"][target_colum].values
    score = model_utils.calculate_metrics(dataset["value"], scorers, y, y_pred, config)
    return score, model_output


# %%
def test_template():
    config = {}
    config["storage"] = "local"
    config["src_dir"] = os.getcwd()
    dataset_obj = json.load(open(os.path.join(os.getcwd(), "datasets_obj/dataset_obj.json")))
    dataset_format = dataset_obj["data_format"]
    print(dataset_format)
    if dataset_format != "list":
        dataset_obj["value"] = LocalResource(key=os.path.join(os.getcwd(), "datasets")).urn
    train(dataset_obj, config)
    run(dataset_obj, config)
    evaluate(dataset_obj, config)
