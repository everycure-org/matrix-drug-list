from kedro.pipeline import Pipeline, pipeline, node
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.generate_ob_list,
                inputs=
                        ["orange-book-products", 
                        "fda_exclusions", 
                        "fda_ob_split_exclusions",
                        "params:desalting_params",
                        "params:name_resolver_params",
                        "params:approval_tag_usa"],
                         
                outputs= "orange_book_list",
                name = "generate-orange-book-list-node"
            ),
            
            node(
                func=nodes.generate_ema_list,
                inputs=
                        ["ema_raw_data_set", 
                        "ema_exclusions", 
                        "ema_split_exclusions",
                        "params:desalting_params",
                        "params:name_resolver_params",
                        "params:approval_tag_europe"],  
                outputs= "ema_list",
                name = "generate-ema-list-node"
            ),

            node(
                func=nodes.generate_pmda_list,
                inputs=
                        ["pmda_raw_data_set", 
                        "pmda_exclusions", 
                        "pmda_split_exclusions",
                        "params:desalting_params",
                        "params:name_resolver_params",
                        "params:approval_tag_japan"],
                         
                outputs= "pmda_list",
                name = "generate-pmda-list-node"
            ),

            node(
                func=nodes.build_drug_list,
                inputs=
                        ["orange_book_list", 
                        "ema_list", 
                        "pmda_list"],
                         
                outputs= "drug_list",
                name = "generate-drug-list-node"
            ),
            node(
                func=nodes.create_standardized_columns_purplebook,
                inputs=[
                    'fda_purple_book_raw_data_set'
                ],
                outputs = 'purple_book_list_standardized',
                name = 'standardize-purplebook'
            ),
            node(
                func=nodes.tag_combination_therapies,
                inputs=[
                    'purple_book_list_standardized',
                    'params:delimiters_purplebook',
                    'params:split_exclusions_purplebook',
                ],
                outputs = 'purple_book_list_with_combination_therapy_tags',
                name = 'tag-combination-therapies-purplebook'
            ),
            node(
                func=nodes.identify_drugs,
                inputs=[
                    'purple_book_list_with_combination_therapy_tags',
                    'params:name_resolver_params'
                ],
                outputs = 'purple_book_list_with_curies',
                name = 'get-curies-purplebook'
            ),
            node(
                func=nodes.add_approval_tags,
                inputs=[
                    'purple_book_list_with_curies',
                    'params:approval_tag_usa'
                ],
                outputs = 'purple_book_list_with_approval_tags',
                name = 'add-approval-tags-purplebook'
            ),
            node(
                func=nodes.enrich_drug_list,
                inputs=['drug_list',
                        'params:enrichment_tags'],
                outputs = 'drug_list_final',
                name = 'drug-list-enrichment'
            )
            

        ]
    )

