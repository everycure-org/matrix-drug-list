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
            #             "params:name_resolver_params",
            #             "params:approval_tag_usa"],
                         
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
            #             "params:name_resolver_params",
            #             "params:approval_tag_europe"],  
            #     outputs= "ema_list",
            #     name = "generate-ema-list-node"
            # ),

            # node(
            #     func=nodes.generate_pmda_list,
            #     inputs=
            #             ["pmda_raw_data_set", 
            #             "pmda_exclusions", 
            #             "pmda_split_exclusions",
            #             "params:desalting_params",
            #             "params:name_resolver_params",
            #             "params:approval_tag_japan"],
                         
            #     outputs= "pmda_list",
            #     name = "generate-pmda-list-node"
            # ),




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
                func=nodes.drop_discontinued_drugs,
                inputs=[
                    'purple_book_list_standardized',
                    "params:marketing_status_column_purplebook",
                    "params:discontinued_marker_purplebook",
                ],
                outputs='purple_book_list_no_discn',
                name="drop-discontinued-drugs-purplebook"
            ),
            node(
                func=nodes.tag_combination_therapies,
                inputs=[
                    'purple_book_list_no_discn',
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
                func = nodes.desalt_drugs,
                inputs=[
                    'purple_book_list_with_ingredients',
                    'params:desalting_params',
                ],
                outputs='purple_book_list_desalted',
                name = 'desalt-list-purplebook',
            ),
            node(
                func=nodes.add_ingredient_ids,
                inputs=[
                    'purple_book_list_desalted',
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

            node(
                func=nodes.return_final_list,
                inputs=[
                    'purple_book_list',
                    'params:drug_list_properties',
                    'params:approval_tag_usa',
                    'params:additional_drug_list_properties_purplebook'
                ],
                outputs = 'purple_book_list_filtered',
                name = 'return_final_list_purplebook'
            ),


            ##########################################################################################
            ##########################################################################################
            ##########################################################################################


            # EMA BUILD
            node(
                func = nodes.preprocess_ema,
                inputs=[
                    'ema-products',
                ],
                outputs = "ema-preprocessed",
                name = "preprocess-ema",
            ),
            node(
                func = nodes.remove_manually_excluded_drugs,
                inputs=[
                    'ema-preprocessed',
                    'params:exclusions_ema',
                    'params:ema_drug_name_column',
                ],
                outputs = "ema-with-exclusions-removed",
                name = "remove-manual-exclusions-ema",
            ),
            node(
                func=nodes.create_standardized_columns,
                inputs=[
                    'ema-with-exclusions-removed',
                    'params:ema_drug_name_column',
                    'params:ema_approval_date_column',
                ],
                outputs = 'ema_list_standardized',
                name = 'standardize-ema'
            ),
            node(
                func=nodes.tag_combination_therapies,
                inputs=[
                    'ema_list_standardized',
                    'params:delimiters_ema',
                    'params:split_exclusions_ema',
                ],
                outputs = 'ema_list_with_combination_therapy_tags',
                name = 'tag-combination-therapies-ema'
            ),

            node(
                func=nodes.add_approval_tags,
                inputs=[
                    'ema_list_with_combination_therapy_tags',
                    'params:approval_tag_europe'
                ],
                outputs = 'ema_list_with_approval_tags',
                name = 'add-approval-tags-ema'
            ),
            node(
                func=nodes.add_ingredients,
                inputs=[
                    'ema_list_with_approval_tags',
                    'params:delimiters_ema'
                ],
                outputs = 'ema_list_with_ingredients',
                name = 'add-ingredients-ema'
            ),
            node(
                func = nodes.desalt_drugs,
                inputs=[
                    'ema_list_with_ingredients',
                    'params:desalting_params',
                ],
                outputs='ema_list_desalted',
                name = 'desalt-list-ema',
            ),
            node(
                func=nodes.add_unlisted_single_ingredients,
                inputs=[
                    'ema_list_desalted',
                ],
                outputs = 'ema_list_with_unlisted_single_ingredients',
                name = 'add-unlisted-single-ingredients-ema'
            ),
            node(
                func=nodes.identify_drugs,
                inputs=[
                    'ema_list_with_unlisted_single_ingredients',
                    'params:name_resolver_params'
                ],
                outputs = 'ema_list_with_curies',
                name = 'get-curies-ema'
            ),
            node(
                func=nodes.add_ingredient_ids,
                inputs=[
                    'ema_list_with_curies',
                    'params:name_resolver_params'
                ],
                outputs = 'ema_list_with_ingredient_ids',
                name = 'add-ingredient-ids-ema'
            ),
            node(
                func=nodes.add_alternate_ids,
                inputs=[
                    'ema_list_with_ingredient_ids'
                ],
                outputs = 'ema_list',
                name = 'add-alternate-ids-ema'
            ),
            node(
                func=nodes.return_final_list,
                inputs=[
                    'ema_list',
                    'params:drug_list_properties',
                    'params:approval_tag_europe',
                    'params:additional_drug_list_properties_ema'
                ],
                outputs = 'ema_list_filtered',
                name = 'return_final_list_ema'
            ),

            ##########################################################################################
            ##########################################################################################
            ##########################################################################################






            # ORANGE BOOK BUILD
            node(
                func = nodes.add_most_permissive_marketing_tags_fda,
                inputs = 'orange-book-products',
                outputs = 'orangebook_list_with_marketing_status',
                name= 'add-marketing-tags-fda',
            ),
            node(
                func=nodes.create_standardized_columns,
                inputs=[
                    'orangebook_list_with_marketing_status',
                    'params:orangebook_drug_name_column',
                    'params:orangebook_approval_date_column',
                ],
                outputs = 'orange_book_list_standardized',
                name = 'standardize-orangebook'
            ),
            # node(
            #     func=nodes.drop_discontinued_drugs,
            #     inputs=[
            #         'orange_book_list_standardized',
            #         "params:marketing_status_column_orangebook",
            #         "params:discontinued_marker_orangebook",
            #     ],
            #     outputs='orange_book_list_no_discn',
            #     name = 'drop-discontinued-drugs-orangebook'
            # ),

            node(
                func=nodes.tag_combination_therapies,
                inputs=[
                    'orange_book_list_standardized',
                    'params:delimiters_orangebook',
                    'params:split_exclusions_orangebook',
                ],
                outputs = 'orange_book_list_with_combination_therapy_tags',
                name = 'tag-combination-therapies-orangebook'
            ),

            node(
                func=nodes.add_approval_tags,
                inputs=[
                    'orange_book_list_with_combination_therapy_tags',
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
                func = nodes.desalt_drugs,
                inputs=[
                    'orange_book_list_with_ingredients',
                    'params:desalting_params',
                ],
                outputs='orange_book_list_desalted',
                name = 'desalt-list-orangebook',
            ),
            node(
                func=nodes.add_unlisted_single_ingredients,
                inputs=[
                    'orange_book_list_desalted',
                ],
                outputs = 'orange_book_list_with_unlisted_single_ingredients',
                name = 'add-unlisted-single-ingredients-orangebook'
            ),
            node(
                func=nodes.identify_drugs,
                inputs=[
                    'orange_book_list_with_unlisted_single_ingredients',
                    'params:name_resolver_params'
                ],
                outputs = 'orange_book_list_with_curies',
                name = 'get-curies-orangebook'
            ),
            node(
                func=nodes.add_ingredient_ids,
                inputs=[
                    'orange_book_list_with_curies',
                    'params:name_resolver_params'
                ],
                outputs = 'orange_book_list_with_ingredient_ids',
                name = 'add-ingredient-ids-orangebook'
            ),

            node(
                func=nodes.check_nameres_llm,
                inputs = [
                    "orange_book_list_with_ingredient_ids",
                    "params:column_names.drug_name",
                    "params:column_names.nameres_first_label",
                    "params:id_correct_incorrect_tag_drug",
                    "params:column_names.llm_true_false_column_drug"
                ],
                outputs = "orange_book_list_with_llm_id_check",
                name = "nameres-auto-qc-drug-fda"
            ),
            node(
                func=nodes.llm_improve_ids,
                inputs = [
                    "orange_book_list_with_llm_id_check",
                    "params:column_names.drug_name",
                    "params:llm_best_id_tag",
                    "params:biolink_type_drug",
                    "params:column_names.disease_id_column",
                    "params:column_names.llm_true_false_column_disease",
                    "params:column_names.llm_improved_id_column",
                ],
                outputs = "dailymed_6",
                name = "llm-id-improvement"
            ),
            node(
                func=nodes.add_alternate_ids,
                inputs=[
                    'orange_book_list_with_ingredient_ids'
                ],
                outputs = 'orange_book_list',
                name = 'add-alternate-ids-orangebook'
            ),
            node(
                func=nodes.return_final_list,
                inputs=[
                    'orange_book_list',
                    'params:drug_list_properties',
                    'params:approval_tag_usa',
                    'params:additional_drug_list_properties_usa',
                ],
                outputs = 'orange_book_list_filtered',
                name = 'return_final_list_orangebook'
            ),


            ##########################################################################################
            ##########################################################################################
            ##########################################################################################

            # # INDIAN APPROVAL LIST
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
                func=nodes.add_approval_tags,
                inputs=[
                    'india_list_with_combination_therapy_tags',
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
                func = nodes.desalt_drugs,
                inputs=[
                    'india_list_with_ingredients',
                    'params:desalting_params',
                ],
                outputs='india_list_desalted',
                name = 'desalt-list-india',
            ),
            node(
                func=nodes.identify_drugs,
                inputs=[
                    'india_list_desalted',
                    'params:name_resolver_params'
                ],
                outputs = 'india_list_with_curies',
                name = 'get-curies-india'
            ),
            node(
                func=nodes.check_nameres_llm,
                inputs = [
                    "india_list_with_curies",
                    "params:column_names.drug_name",
                    "params:column_names.nameres_first_label",
                    "params:id_correct_incorrect_tag_drug",
                    "params:column_names.llm_true_false_column_drug"
                ],
                outputs = "india_list_with_llm_id_check",
                name = "nameres-auto-qc-drug-india"
            ),
            # node(
            #     func=nodes.llm_improve_ids,
            #     inputs = [
            #         "india_list_with_llm_id_check",
            #         "params:column_names.drug_name",
            #         "params:llm_best_id_tag",
            #         "params:biolink_type_drug",
            #         "params:column_names.disease_id_column",
            #         "params:column_names.llm_true_false_column_disease",
            #         "params:column_names.llm_improved_id_column",
            #     ],
            #     outputs = "india_list_improved_ids",
            #     name = "llm-id-improvement-india"
            # ),
           
            # node(
            #     func=nodes.add_ingredients,
            #     inputs=[
            #         'india_list_with_approval_tags',
            #         'params:delimiters_india'
            #     ],
            #     outputs = 'india_list_with_ingredients',
            #     name = 'add-ingredients-india'
            # ),
            # node(
            #     func=nodes.add_ingredient_ids,
            #     inputs=[
            #         'india_list_with_ingredients',
            #         'params:name_resolver_params'
            #     ],
            #     outputs = 'india_list_with_ingredient_ids',
            #     name = 'add-ingredient-ids-india'
            # ),
            # node(
            #     func=nodes.add_alternate_ids,
            #     inputs=[
            #         'india_list_with_ingredient_ids'
            #     ],
            #     outputs = 'india_list',
            #     name = 'add-alternate-ids-india'
            # ),
 
 
            ##########################################################################################
            ##########################################################################################
            ##########################################################################################


            # RUSSIA
            node(
                func = nodes.translate_dataframe_columns,
                inputs=[
                    'russia_base_list',
                    'params:in_language_russia',
                    'params:out_language_russia',
                ],
                outputs = 'russia_list_translated',
                name = 'translate-russia-list'
            ),
            node(
                func=nodes.create_standardized_columns,
                inputs=[
                    "russia_list_translated",
                    "params:russia_drug_column",
                    "params:russia_approval_date_column",   
                ],
                outputs = "russia_list_standardized",
                name = "standardize-columns-russia"
            ),
            node(
                func=nodes.tag_combination_therapies,
                inputs=[
                    'russia_list_standardized',
                    'params:delimiters_russia',
                    'params:split_exclusions_russia',
                ],
                outputs = 'russia_list_with_combination_therapy_tags',
                name = 'tag-combination-therapies-russia'
            ),
            node(
                func=nodes.add_approval_tags,
                inputs=[
                    'russia_list_with_combination_therapy_tags',
                    'params:approval_tag_russia'
                ],
                outputs = 'russia_list_with_approval_tags',
                name = 'add-approval-tags-russia'
            ),
            node(
                func=nodes.add_ingredients,
                inputs=[
                    'russia_list_with_approval_tags',
                    'params:delimiters_russia'
                ],
                outputs = 'russia_list_with_ingredients',
                name = 'add-ingredients-russia'
            ),
            node(
                func = nodes.desalt_drugs,
                inputs=[
                    'russia_list_with_ingredients',
                    'params:desalting_params',
                ],
                outputs='russia_list_desalted',
                name = 'desalt-list-russia',
            ),
            node(
                func=nodes.identify_drugs,
                inputs=[
                    'russia_list_desalted',
                    'params:name_resolver_params'
                ],
                outputs = 'russia_list_with_curies',
                name = 'get-curies-russia'
            ),
            
            node(
                func=nodes.add_ingredient_ids,
                inputs=[
                    'russia_list_with_curies',
                    'params:name_resolver_params'
                ],
                outputs = 'russia_list_with_ingredient_ids',
                name = 'add-ingredient-ids-russia'
            ),
            node(
                func=nodes.check_nameres_llm,
                inputs = [
                    "russia_list_with_ingredient_ids",
                    "params:column_names.drug_name",
                    "params:column_names.nameres_first_label",
                    "params:id_correct_incorrect_tag_drug",
                    "params:column_names.llm_true_false_column_drug"
                ],
                outputs = "russia_list_with_llm_id_check",
                name = "nameres-auto-qc-drug-russia"
            ),

            node(
                func=nodes.llm_improve_ids,
                inputs = [
                    "russia_list_with_llm_id_check",
                    "params:column_names.drug_name",
                    "params:llm_best_id_tag",
                    "params:biolink_type_drug",
                    "params:column_names.disease_id_column",
                    "params:column_names.llm_true_false_column_disease",
                    "params:column_names.llm_improved_id_column",
                ],
                outputs = "russia_list_improved_ids",
                name = "llm-id-improvement-russia"
            ),
            ##########################################################################################
            ##########################################################################################
            ##########################################################################################

            # JAPANESE APPROVAL LIST
            node(
                func = nodes.preprocess_pmda,
                inputs=[
                    'pmda-products'
                ],
                outputs = 'pmda-preprocessed',
                name = 'preprocess-pmda'

            ),
            node(
                func = nodes.remove_manually_excluded_drugs,
                inputs=[
                    'pmda-preprocessed',
                    'params:exclusions_pmda',
                    'params:pmda_drug_name_column',
                ],
                outputs = "pmda-with-exclusions-removed",
                name = "remove-manual-exclusions-pmda",
            ),
            node(
                func=nodes.create_standardized_columns,
                inputs=[
                    'pmda-with-exclusions-removed',
                    'params:pmda_drug_name_column',
                    'params:pmda_approval_date_column',
                ],
                outputs = 'pmda_list_standardized',
                name = 'standardize-pmda'
            ),
            # node(
            #     func=nodes.reformat_ingredients_with_ollama,
            #     inputs=[
            #         'pmda_list_standardized',
            #         'params:name_reformatting_prompt'
            #     ],
            #     outputs='pmda_list_names_reformatted',
            #     name='reformat-ingredient-names-pmda',
            # ),
            node(
                func=nodes.tag_combination_therapies,
                inputs=[
                    'pmda_list_standardized',
                    'params:delimiters_pmda',
                    'params:split_exclusions_pmda',
                ],
                outputs = 'pmda_list_with_combination_therapy_tags',
                name = 'tag-combination-therapies-pmda'
            ),

            node(
                func=nodes.add_approval_tags,
                inputs=[
                    'pmda_list_with_combination_therapy_tags',
                    'params:approval_tag_japan'
                ],
                outputs = 'pmda_list_with_approval_tags',
                name = 'add-approval-tags-pmda'
            ),
            node(
                func=nodes.add_ingredients,
                inputs=[
                    'pmda_list_with_approval_tags',
                    'params:delimiters_pmda'
                ],
                outputs = 'pmda_list_with_ingredients',
                name = 'add-ingredients-pmda'
            ),
            node(
                func=nodes.add_unlisted_single_ingredients,
                inputs=[
                    'pmda_list_with_ingredients',
                ],
                outputs = 'pmda_list_with_unlisted_single_ingredients',
                name = 'add-unlisted-single-ingredients-pmda'
            ),
            node(
                func=nodes.identify_drugs,
                inputs=[
                    'pmda_list_with_unlisted_single_ingredients',
                    'params:name_resolver_params'
                ],
                outputs = 'pmda_list_with_curies',
                name = 'get-curies-pmda'
            ),
            node(
                func=nodes.add_ingredient_ids,
                inputs=[
                    'pmda_list_with_curies',
                    'params:name_resolver_params'
                ],
                outputs = 'pmda_list_with_ingredient_ids',
                name = 'add-ingredient-ids-pmda'
            ),
            node(
                func=nodes.add_alternate_ids,
                inputs=[
                    'pmda_list_with_ingredient_ids'
                ],
                outputs = 'pmda_list',
                name = 'add-alternate-ids-pmda'
            ),
            node(
                func=nodes.return_final_list,
                inputs=[
                    'pmda_list',
                    'params:drug_list_properties',
                    'params:approval_tag_japan',
                    'params:additional_drug_list_properties_pmda'
                ],
                outputs = 'pmda_list_filtered',
                name = 'return_final_list_pmda'
            ),
            

            # node(
            # func=nodes.build_drug_list,
            # inputs=
            #         ["orange_book_list", 
            #         "ema_list", 
            #         "pmda_list"],
                        
            # outputs= "drug_list",
            # name = "generate-drug-list-node"
            # ),













            node(
                func=nodes.merge_all_drug_lists,
                inputs=[
                        'pmda_list_filtered',
                        'ema_list_filtered',
                        'orange_book_list_filtered',
                        'purple_book_list_filtered',
                ],
                outputs='drug_list_merged',
                name='merge-drug-lists'
            ),

            ##########################################################################################
            ##########################################################################################
            ##########################################################################################




            # DRUG LIST CATEGORIZATION TAGS
            node(
                func=nodes.enrich_drug_list,
                inputs=['drug_list_merged',
                        'params:enrichment_tags',
                        'params:llm_to_use'],
                outputs = 'drug_list_with_category_tags',
                name = 'drug-list-enrichment'
            ),
            node(
                func=nodes.add_approval_false_tags,
                inputs=[
                    'drug_list_with_category_tags',
                    'params:approval_tag_usa',
                ],
                outputs='drug_list_corrected_approval_tags_usa',
                name = 'correct-tags-usa'
            ),
            node(
                func=nodes.add_approval_false_tags,
                inputs=[
                    'drug_list_corrected_approval_tags_usa',
                    'params:approval_tag_europe',
                ],
                outputs='drug_list_corrected_approval_tags_europe',
                name = 'correct-tags-europe'

            ),
            node(
                func=nodes.add_approval_false_tags,
                inputs=[
                    'drug_list_corrected_approval_tags_europe',
                    'params:approval_tag_japan',
                ],
                outputs='drug_list_final',
                name = 'correct-tags-japan',
            ),
            
            # ADD SMILES STRINGS WHEN APPLICABLE
            node(
                func=nodes.add_SMILES_strings,
                inputs=[
                    "drug_list_final",
                ],
                outputs="drug_list_with_smiles",
                name = "add-smiles-to-list"
            )

        ]
    )

