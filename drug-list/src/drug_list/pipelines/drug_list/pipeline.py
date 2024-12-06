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
            
            # PURPLE BOOK BUILD
            node(
                func=nodes.create_standardized_columns,
                inputs=[
                    'fda_purple_book_raw_data_set',
                    'params:purplebook_drug_name_column',
                    'params:purplebook_approval_date_column',
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
                func=nodes.add_ingredients,
                inputs=[
                    'purple_book_list_with_approval_tags',
                    'params:delimiters_purplebook'
                ],
                outputs = 'purple_book_list_with_ingredients',
                name = 'add-ingredients-purplebook'
            ),
            node(
                func=nodes.add_ingredient_ids,
                inputs=[
                    'purple_book_list_with_ingredients',
                    'params:name_resolver_params'
                ],
                outputs = 'purple_book_list_with_ingredient_ids',
                name = 'add-ingredient-ids-purplebook'
            ),
            node(
                func=nodes.add_alternate_ids,
                inputs=[
                    'purple_book_list_with_ingredient_ids'
                ],
                outputs = 'purple_book_list',
                name = 'add-alternate-ids-purplebook'
            ),

            # ORANGE BOOK BUILD
            node(
                func=nodes.create_standardized_columns,
                inputs=[
                    'orange-book-products',
                    'params:orangebook_drug_name_column',
                    'params:orangebook_approval_date_column',
                ],
                outputs = 'orange_book_list_standardized',
                name = 'standardize-orangebook'
            ),
            node(
                func=nodes.tag_combination_therapies,
                inputs=[
                    'orange_book_list_standardized',
                    'params:delimiters_orangebook',
                    'params:split_exclusions_orangebook',
                ],
                outputs = 'purple_book_list_with_combination_therapy_tags',
                name = 'tag-combination-therapies-orangebook'
            ),
            node(
                func=nodes.identify_drugs,
                inputs=[
                    'orange_book_list_with_combination_therapy_tags',
                    'params:name_resolver_params'
                ],
                outputs = 'orange_book_list_with_curies',
                name = 'get-curies-orangebook'
            ),
            node(
                func=nodes.add_approval_tags,
                inputs=[
                    'orange_book_list_with_curies',
                    'params:approval_tag_usa'
                ],
                outputs = 'orange_book_list_with_approval_tags',
                name = 'add-approval-tags-orangebook'
            ),
            node(
                func=nodes.add_ingredients,
                inputs=[
                    'orange_book_list_with_approval_tags',
                    'params:delimiters_orangebook'
                ],
                outputs = 'orange_book_list_with_ingredients',
                name = 'add-ingredients-orangebook'
            ),
            node(
                func=nodes.add_ingredient_ids,
                inputs=[
                    'orange_book_list_with_ingredients',
                    'params:name_resolver_params'
                ],
                outputs = 'orange_book_list_with_ingredient_ids',
                name = 'add-ingredient-ids-purplebook'
            ),
            node(
                func=nodes.add_alternate_ids,
                inputs=[
                    'orange_book_list_with_ingredient_ids'
                ],
                outputs = 'orange_book_list',
                name = 'add-alternate-ids-orangebook'
            ),


            # INDIAN APPROVAL LIST
            node(
                func=nodes.create_standardized_columns,
                inputs=[
                    'indian_drug_approvals_raw_data_set',
                    'params:india_drug_name_column',
                    'params:india_approval_date_column',
                ],
                outputs= 'indian_drugs_standardized',
                name = 'standardize-indian-drug-list-raw'
            ),
            node(
                func=nodes.tag_combination_therapies,
                inputs=[
                    'indian_drugs_standardized',
                    'params:delimiters_india',
                    'params:split_exclusions_india',
                ],
                outputs = 'india_list_with_combination_therapy_tags',
                name = 'tag-combination-therapies-india'
            ),
            node(
                func=nodes.identify_drugs,
                inputs=[
                    'india_list_with_combination_therapy_tags',
                    'params:name_resolver_params'
                ],
                outputs = 'india_list_with_curies',
                name = 'get-curies-india'
            ),
            node(
                func=nodes.add_approval_tags,
                inputs=[
                    'india_list_with_curies',
                    'params:approval_tag_india'
                ],
                outputs = 'india_list_with_approval_tags',
                name = 'add-approval-tags-india'
            ),
            node(
                func=nodes.add_ingredients,
                inputs=[
                    'india_list_with_approval_tags',
                    'params:delimiters_india'
                ],
                outputs = 'india_list_with_ingredients',
                name = 'add-ingredients-india'
            ),
            node(
                func=nodes.add_ingredient_ids,
                inputs=[
                    'india_list_with_ingredients',
                    'params:name_resolver_params'
                ],
                outputs = 'india_list_with_ingredient_ids',
                name = 'add-ingredient-ids-india'
            ),
            node(
                func=nodes.add_alternate_ids,
                inputs=[
                    'india_list_with_ingredient_ids'
                ],
                outputs = 'india_list',
                name = 'add-alternate-ids-india'
            ),


            # DRUG LIST CATEGORIZATION TAGS
            node(
                func=nodes.enrich_drug_list,
                inputs=['drug_list',
                        'params:enrichment_tags'],
                outputs = 'drug_list_final',
                name = 'drug-list-enrichment'
            )
            

        ]
    )

