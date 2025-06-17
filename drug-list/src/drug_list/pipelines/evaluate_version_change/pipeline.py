from kedro.pipeline import Pipeline, pipeline, node
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.store_previous_version,
                inputs="purple_book_list_filtered",
                outputs="purple_book_prev",
                name="store-fda-pb"
            ),
            node(
                func=nodes.store_previous_version,
                inputs="orange_book_list_filtered",
                outputs="orange_book_prev",
                name = "store-fda-ob"
            ),
            node(
                func=nodes.store_previous_version,
                inputs="ema_list_filtered",
                outputs="ema_prev",
                name = "store-ema"
            ),
            node(
                func=nodes.store_previous_version,
                inputs="pmda_list_filtered",
                outputs="pmda_prev",
                name = "store-pmda"
            ),
            node(
                func=nodes.store_previous_version,
                inputs="india_list_filtered",
                outputs="india_prev",
                name = "store-india"
            ),
            node(
                func=nodes.store_previous_version,
                inputs="russia_list_filtered",
                outputs="russia_prev",
                name = "store-russia"
            ),
        ]
    )

