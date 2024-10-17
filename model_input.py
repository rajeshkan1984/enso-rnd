# %%
from datetime import datetime


# %%
def model_input(config=None, **kwargs):
    result = {"dataset": {
        "name": datetime.utcnow().isoformat(),
        "data_format": "csv",
        "value": kwargs["document"][0]["metadata"]["properties"]["file_metadata"]["file_path"]
    }}
    result.update(kwargs)
    return result
