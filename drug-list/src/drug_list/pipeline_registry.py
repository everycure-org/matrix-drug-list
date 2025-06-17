"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline


from drug_list.pipelines.drug_list.pipeline import (
    create_pipeline as create_drug_list_pipeline,
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {}
    pipelines["__default__"] = create_drug_list_pipeline()
    
    return pipelines
