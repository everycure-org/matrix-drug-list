from kedro.pipeline import Pipeline, pipeline, node

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=nodes.generate_ob_list,
            #     inputs=
            #             ["orange-book-products", 
            #             "fda_exclusions", 
            #             "fda_ob_split_exclusions",
            #             "params:desalting_params",
            #             "params:name_resolver_params"],
                         
            #     outputs= "orange_book_list",
            #     name = "generate-orange-book-list-node"
            # ),
            
            # node(
            #     func=nodes.generate_ema_list,
            #     inputs=
            #             ["ema_raw_data_set", 
            #             "ema_exclusions", 
            #             "ema_split_exclusions",
            #             "params:desalting_params",
            #             "params:name_resolver_params"],
            #             #"ema_raw_data_set",
            #             #"ema_exclusions",
            #             #"ema_split_exclusions",
            #             #"pmda_raw_data_set",
            #             #"pmda_exclusions",
            #             #"pmda_split_exclusions"],
                         
            #     outputs= "ema_list",
            #     name = "generate-ema-list-node"
            # ),

            node(
                func=nodes.generate_pmda_list,
                inputs=
                        ["pmda_raw_data_set", 
                        "pmda_exclusions", 
                        "pmda_split_exclusions",
                        "params:desalting_params",
                        "params:name_resolver_params"],
                         
                outputs= "pmda_list",
                name = "generate-pmda-list-node"
            ),


        ]
    )

