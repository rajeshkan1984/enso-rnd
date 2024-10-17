import os
import pandas as pd
import numpy as np
import traceback
import pickle

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
    
    
class Lta_preprocess:
    
    def __init__(self,df):
        self.df = df
    
    def dob_transform(self):
        # date of birth transformation
        import re
        existing_column = "PATIENT_DOB"
        new_column = "AGE_TYPE"
        date_patterns = [
            re.compile(r'^(3[01]|[12][0-9]|0?[1-9])[ ./-](1[0-2]|0?[1-9])[ ./-][0-9]{4}$'),  # DD-MM-YYYY
            re.compile(r'^(1[0-2]|0?[1-9])[ ./-](3[01]|[12][0-9]|0?[1-9])[ ./-][0-9]{4}$'),  # MM-DD-YYYY
            re.compile(r'^[0-9]{4}[ ./-](1[0-2]|0?[1-9])[ ./-](3[01]|[12][0-9]|0?[1-9])$'),  # YYYY-MM-DD
            re.compile(r'^[0-9]{4}(1[0-2]|0?[1-9])(3[01]|[12][0-9]|0?[1-9])$')  # YYYYMMDD
        ]

        
        def is_valid_date_format(date_str):
            return any(pattern.match(date_str) for pattern in date_patterns)

        
        if self.df[existing_column].isnull().any() or (self.df[existing_column] == '').any():
            raise ValueError(
                f"The column '{existing_column}' contains blank values. Please provide a complete Date column.")

    
        if not self.df[existing_column].apply(lambda x: is_valid_date_format(str(x))).all():
            raise ValueError(
                f"The column '{existing_column}' contains invalid date format. Please provide a valid Date column.")

        try:
            temp_col = pd.to_datetime(self.df[existing_column], errors='raise')
        except Exception as e:
            raise ValueError(
                f"The column '{existing_column}' contains invalid date formats. Please provide a valid Date column.") from e

        today = pd.to_datetime('today')
        if (temp_col > today).any():
            raise ValueError(
                f"The column '{existing_column}' contains future dates. Please provide a valid Date column.")

        age = today.year - pd.to_datetime(self.df[existing_column]).dt.year

        # Define conditions and choices for age categories
        conditions = [
            (age <= 18),
            (age >= 66),
            (age > 18) & (age < 66)
        ]
        choices = ['IS_MINOR', 'IS_SENIOR_CITIZEN', 'IS_ADULT']

        if new_column:
            self.df[new_column] = np.select(conditions, choices, default='UNKNOWN')
        else:
            self.df[existing_column] = np.select(conditions, choices, default='UNKNOWN')

        return self.df.columns.tolist()

    def set_datatype(self):
        # set datatype
        columns = {"PATIENT_GENDER": "str", "PROCEDURE": "str", "PROCEDURE_MODIFIER_COMBO": "str",
                   "LINE_CHARGE": "float", "DIAGNOSIS_COMBO": "str", "RENDERING_PROVIDER_NPI": "str",
                   "EMERGENCY": "str", "RENDERING_CLAIM_SOURCE_SPECIALTY_TAXONOMY": "str",
                   "FACILITY_ENTITY_TYPE_CODE": "str", "MEMBER_ADR_ZIP": "str", "TYPE_BILL": "str",
                   "VENDOR_SOURCE_CODE": "str", "FACILITY_TYPE_CODE": "str", "BILLING_PROVIDER_NPI": "str",
                   "LINE_ITEM_PROVIDER_PAYMENT_AMOUNT": "float", "UNITS": "int", "CLAIM_STATUS_CODE": "int",
                   "AGE_TYPE": "str"}

        cols = columns
        if not cols:
            raise ValueError("The parameter cannot be empty. Please provide a valid input.")
        valid_types = {
            'str': str,
            'int': int,
            'int32': 'int32',
            'int64': 'int64',
            'float': float,
            'float32': 'float32',
            'float64': 'float64'
        }
        col_type = {}
        for col, dtype in cols.items():
            if col not in self.df.columns:
                raise ValueError(f"Feature '{col}' not found in the DataFrame.")

            dtype = dtype.lower()
            if dtype not in valid_types:
                raise ValueError(
                    f"Invalid data type '{dtype}' specified for Feature '{col}'. Valid types are 'str', 'int', 'float', 'int32', 'int64', 'float32', 'float64'.")

            col_type[col] = valid_types[dtype]
        # for k, v in cols.items():
        #     if v == 'str':
        #         col_type[k] = str
        #     elif v == 'int':
        #         col_type[k] = int
        #     elif v == 'float':
        #         col_type[k] = float
        self.df = self.df.astype(col_type)
        return self.df
    
    def filter_column(self):
        # filter columns
        cols = ["CLAIM_NUMBER","PLACE_SERVICE", "PROCEDURE", "PROCEDURE_MODIFIER_COMBO", "LINE_CHARGE", "DIAGNOSIS_COMBO",
                "RENDERING_PROVIDER_NPI", "EMERGENCY", "RENDERING_CLAIM_SOURCE_SPECIALTY_TAXONOMY",
                "FACILITY_ENTITY_TYPE_CODE", "MEMBER_ADR_ZIP", "TYPE_BILL", "VENDOR_SOURCE_CODE", "FACILITY_TYPE_CODE",
                "BILLING_PROVIDER_NPI", "BILLING_ADR_ZIP", "CLAIM_STATUS_CODE", "PATIENT_GENDER", "AGE_TYPE"]

        self.df = self.df[cols]
        return self.df
    
    def ohe(self):
        # one hot encoder
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        cols = ["PATIENT_GENDER", "AGE_TYPE"]
    
        if isinstance(cols, list):
            if len(cols) == 0:
                raise Exception("The Parameter Cannot Be Empty. Please Provide A Valid Input.")
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            ohe.fit(self.df[cols])
            print(ohe.get_feature_names_out(cols))
            encoded_columns = ohe.get_feature_names_out(cols)
            encoded_data = pd.DataFrame(ohe.transform(self.df[cols]).toarray(), columns=encoded_columns).reset_index(drop=True)
            updated_df = pd.concat([self.df.reset_index(drop=True), encoded_data], axis=1)
            updated_df.drop(cols, inplace=True, axis=1)
        self.df = updated_df
        return self.df
    
    def binary_encoder(self):
        # binary encoder
        from category_encoders import BinaryEncoder
        cols = ["PROCEDURE_MODIFIER_COMBO", "PLACE_SERVICE", "PROCEDURE", "DIAGNOSIS_COMBO", "RENDERING_PROVIDER_NPI",
                "EMERGENCY", "RENDERING_CLAIM_SOURCE_SPECIALTY_TAXONOMY", "FACILITY_ENTITY_TYPE_CODE", "MEMBER_ADR_ZIP",
                "TYPE_BILL", "VENDOR_SOURCE_CODE", "FACILITY_TYPE_CODE", "BILLING_PROVIDER_NPI", "BILLING_ADR_ZIP"]

        binary_encoder = BinaryEncoder(cols=cols)
        df_encoded = binary_encoder.fit_transform(self.df[cols])
        print(df_encoded.columns.tolist())

        df_final = pd.concat([self.df.drop(columns=cols).reset_index(drop=True), df_encoded.reset_index(drop=True)],
                             axis=1)
        self.df = df_final
        return self.df

    def run(self):
        self.dob_transform()
        self.set_datatype()
        self.filter_column()
        self.ohe()
        self.binary_encoder()
        return self.df

def lta_preprocessing(config=None, **objects):
    file_path = objects['document'][0]['metadata']['properties']['file_metadata']['file_path']
    doc_id = objects['document'][0]['doc_id']
    solution_id = objects['document'][0]['solution_id']
    local_file_path = Util.download_minio_file(file_path)
    df = Util.read_file(local_file_path)

    df = Lta_preprocess(df).run()

    local_file_path = f"/tmp/{doc_id}.csv"
    df.to_csv(local_file_path, index=False)
    minio_path = Util.upload_to_minio(solution_id, local_file_path, minio_path=file_path)
    return objects