"""
This is a boilerplate pipeline 'generate_orange_book_list'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.generate_ob_list,
                inputs=["orange-book-products", "fda_exclusions", "fda_ob_split_exclusions" ], 
                outputs="orange_book_list",
                name = "generate-orange-book-list-node"
            ),
        ]
    )