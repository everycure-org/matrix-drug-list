from kedro.pipeline import Pipeline, pipeline, node
from .nodes import generate_ob_list

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
            func=generate_ob_list,
            inputs=["orange-book-products", "fda_exclusions", "fda_ob_split_exclusions" ], 
            outputs="orange_book_list",
            name = "generate-orange-book-list-node"
            ),
        ]
    )