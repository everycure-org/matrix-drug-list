import pandas as pd
pd.options.mode.chained_assignment = None
import difflib as dl
import re
import requests
from io import StringIO
from typing import List
from tqdm import tqdm
import json
import os
from typing import List, Dict, Optional, Any
from openai import OpenAI
import numpy as np
from functools import cache
import asyncio
from googletrans import Translator

import ast
import time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

testing = False
limit = 1000 # limit for testing full pipeline with limited number of names per list



#######################################
##### LIST-SPECIFIC UTILITIES ######
#######################################

def preprocess_ema(rawdata: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        rawdata(pd.DataFrame): raw EMA data
    Returns:
        pd.DataFrame: a raw list of all of the drugs in the EMA book with some processing done to remove problematic rows.
    """
    indices_to_drop = []
    for idx, row in rawdata.iterrows():
        if (row['Category'] != "Human"):
            indices_to_drop.append(idx)
    new_df = rawdata.drop(indices_to_drop)
    new_df.rename(columns={"Authorisation status": "marketing_status_europe"}, inplace=True)
    new_names = []
    for idx, row in new_df.iterrows():
        drugname = row['International non-proprietary name (INN) / common name']
        if type(drugname)==float:
            new_names.append(drugname)
        else:
            new_names.append(drugname.upper())
    new_df['International non-proprietary name (INN) / common name'] = new_names
    return new_df

def preprocess_pmda(rawdata:pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        rawdata(pd.DataFrame): raw PMDA data
    Returns:
        pd.DataFrame: processed PMDA data based on existing problems.
    """
    new_names = []
    name_label = "Active Ingredient (underlined: new active ingredient)"
    for idx, row in tqdm(rawdata.iterrows(), total=len(rawdata), desc="cleaning PMDA list"):
        item = row[name_label]
        try:
            item = item.upper()
            item = item.strip().replace("A COMBINATION DRUG OF", "").strip(' ')
            if "(" in item:
                item = item.replace("(1)","").replace("(2)","").replace("(3)","").replace("(4)","").replace("(5)","").replace("(6)","").replace("(7)","").replace("(8)","")\
                .replace("(9)","").replace("(10)","").replace("(11)","").replace("(12)","").replace("(13)","").replace("(14)","").replace("(15)","").replace("(16)","")\
                .replace("(17)","").replace("(18)","").replace("(19)","").replace("(20)","").replace("(20)","").replace("(21)","").replace("1)","").replace("2)","").replace("5)","")

            item = item.replace("1)", "").replace("2)", "").replace("5)","")

            item = item.replace("\n", " ")

            if type(item)!=float and (("," in item) or ("/" in item) or (" AND " in item)) and item not in exclusions:
                item= item.replace(",","; ").replace(" AND ", "; ").replace("/","; ").replace(";;",";").replace(";  ", "; ").replace("  ;", ";").replace(" ;",";").strip()
            new_names.append(item)
        except:
            print("encountered problem with ", item)
            new_names.append("error")
    rawdata[name_label] = new_names
    return rawdata

def add_marketing_status_tags_ema(rawdata: pd.DataFrame) -> pd.DataFrame:
    return None

def getAllStatuses(orangebook: pd.DataFrame, item: str) -> list[str]:
    """
    Args:
        orangebook (pd.DataFrame): orange book raw data
        item (str): name of the drug whose statuses are to be returned

    Returns:
        list[str]: availability status of all drug formulations for named drug in the United States

    """

    indices = [i for i, x in enumerate(orangebook['Ingredient']) if x == item]
    return list(orangebook['Type'][indices])

def getMostPermissiveStatus(statusList: list[str]) -> str:
    """
    Args:
        statusList (list[str]): list of statuses for a particular chemical entity

    Returns:
        str: The most permissive availability status for the chemical entity in the US (over the counter > RX > DISCN) or "UNSURE" if not clear.

    """
    if "OTC" in statusList:
        return "OTC"
    elif "RX" in statusList:
        return "RX"
    elif "DISCN" in statusList:
        return "DISCONTINUED"
    return "UNSURE"


#########################################
##### HELPER FUNCTIONS ##################
#########################################

def isBasicSaltOrMetalOxide(inString: str, desalting_params: dict) -> bool:
    """
    Args:
        inString (str): name of drug
        desalting_params (dict): parameters for desalting, including inactive cation and anion names and other terms e.g. hydrate, dibasic, etc.
    Returns:
        bool: if the drug is a simple salt or metal oxide (sodium chloride, potassium phosphate, etc.), return TRUE. Otherwise, return FALSE
    """
    components = inString.strip().split()
    for item in components:
        item = item.replace(';', '').replace(',','')
        if not item in desalting_params['basic_cations'] and not item in desalting_params['basic_anions'] and not item in desalting_params['other_identifiers']:
            return False 
    return True

def removeCationsAnionsAndBasicTerms(ingredientString, desalting_params):
    """
    Args:
        ingredientString (str): name of drug ingredient
        desalting_params (dict): parameters for desalting, including inactive cation and anion names and other terms e.g. hydrate, dibasic, etc.
    Returns:
        str: cleaned ingredient name with salts removed. Ideally this will be just the active moiety. The full salt name is returned if it is
             a simple salt, e.g. sodium chloride or potassium phosphate.
    """
    if not isBasicSaltOrMetalOxide(ingredientString, desalting_params):
        components = ingredientString.strip().split()
        for ind,i in enumerate(components):
            if i in desalting_params['basic_anions'] or i in desalting_params['basic_cations'] or i in desalting_params['other_identifiers']:
                components[ind] = ''
        newString = ''
        for i in components:
            newString = newString + i + " "
        newString = newString[:-1]
        return newString
    return ingredientString.strip()

def remove_illegal_characters_openpyxl(input_list: pd.DataFrame):
    return input_list.applymap(lambda x: x.encode('unicode_escape').
                 decode('utf-8') if isinstance(x, str) else x)

def is_combination_therapy(item: str, delimiters: list[str], exclusions: list[str]) -> bool:
    if type(item)==float:
        return False
    if item in exclusions:
        return False
    for i in delimiters:
        if i in item:
            return True
    return False

def preferRXCUI(curieList:list[str], labelList:list[str]) -> tuple:
    """
    Args: 
        curieList (list[str]): list of Curie IDs
        labelList (list[str]): list of labels for respective Curie IDs

    Returns:
        tuple: first Curie ID that is in RXCUI and associated label, or just first curie and label if no RXCUI.

    """

    for idx, item in enumerate(curieList):
        if "RXCUI" in item:
            return item, labelList[idx]
    return curieList[0], labelList[0]  

def identify(name: str, params: dict, prefer_rxcui: bool):
    """
    Args:
        name (str): string to be identified
        params (tuple): name resolver parameters to feed into get request
    
    Returns:
        resolvedName (list[str]): IDs most closely matching string.
        resolvedLabel (list[str]): labels associated with respective resolvedName items.

    """
    if type(name) == float:
        return ['error'], ['error']
    
    itemRequest = (params['url']+
                   params['service']+
                   '?string='+
                   name+
                   '&autocomplete='+
                   str(params['autocomplete_setting']).lower()+
                   '&offset='+
                   str(params['offset'])+
                   '&limit='+
                   str(params['id_limit'])+
                   "&biolink_type="+
                   params['biolink_type'])
    success = False
    failedCounts = 0

    if testing:
        return ["test"], ["test"]
    
    while not success:
        try:
            returned = (pd.read_json(StringIO(requests.get(itemRequest).text)))
            resolvedName = returned.curie
            resolvedLabel = returned.label
            success = True
        except:
            #print('name resolver error')
            failedCounts += 1
        
        if failedCounts >= 5:
            return ["Error"], ["Error"]
    if prefer_rxcui:
        return preferRXCUI(resolvedName, resolvedLabel)
    return resolvedName[0], resolvedLabel[0]

def multi_split(inString, delims):
    result = [inString]
    for delim in delims:
        result = [substr for s in result for substr in s.split(delim)]
    return [s for s in result if s]

def query_ollama(
    prompt: str,
    model: str = "gemma3:1b",
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    url: str = "http://localhost:11434"
    ) -> Dict[str, Any]:
    """
    Send a prompt to a locally running Ollama model and return its response.
    
    Args:
        prompt (str): The main prompt/question to send to the model
        model (str): Name of the model to use (default is Llama-3.3-70B)
        system_prompt (str, optional): System prompt to set context/behavior
        temperature (float): Controls randomness (0.0 to 1.0)
        max_tokens (int, optional): Maximum number of tokens to generate
        top_p (float, optional): Nucleus sampling parameter
        top_k (int, optional): Top-k sampling parameter
        url (str): Base URL for the Ollama API
        
    Returns:
        Dict[str, Any]: Response from the model containing generated text and metadata
        
    Raises:
        requests.exceptions.RequestException: If API request fails
        json.JSONDecodeError: If response parsing fails
    """
    
    # Construct request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    # Add optional parameters if provided
    if system_prompt:
        payload["system"] = system_prompt
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    if top_p:
        payload["options"]["top_p"] = top_p
    if top_k:
        payload["options"]["top_k"] = top_k

    try:
        # Send POST request to Ollama API
        response = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to communicate with Ollama API: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Ollama API response: {str(e)}")

def add_row (original_list: pd.DataFrame, columns: dict) -> pd.DataFrame:
    original_list = pd.concat([original_list, pd.DataFrame(columns, index = [0])], ignore_index=True)
    return original_list

def string_to_list(input_string):
    """
    Convert a string representation of a list into an actual Python list of strings.
    
    Args:
        input_string (str): String representation of a list (e.g., "['item1','item2','item3']")
        
    Returns:
        list: A list of strings
    
    Example:
        >>> string_to_list("['item1','item2','item3']")
        ['item1', 'item2', 'item3']
    """
    cleaned_string = input_string.strip('[]')
    items = [item.strip().strip("'").strip('"') for item in cleaned_string.split(',')]
    items = [item for item in items if item]
    return items

def normalize(item: str):
    if testing:
        return ["test"], ["test"]
    item_request = f"https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes?curie={item}&conflate=true&drug_chemical_conflate=true&description=false&individual_types=false"    
    success = False
    failedCounts = 0
    while not success:
        try:
            response = requests.get(item_request)
            output = json.loads(response.text)
            primary_key = list(output.keys())[0]
            label = output[primary_key]['id']['label']
            alternate_ids = output[item]['equivalent_identifiers']
            returned_ids = list(item['identifier'] for item in alternate_ids)
            success = True
        except Exception as e:
            print(e)
            #print('name resolver error')
            failedCounts += 1
        if failedCounts >= 5:
            return "Error", "Error"
    return returned_ids, label

def generate_tag_localLlama(drug_list:List, model_params:Dict) -> List:
    tag_list = []
    for drug in tqdm(drug_list):
        output = query_ollama(f"{model_params.get('prompt')} drug: {drug}")
        tag_list.append(output["response"])
    return tag_list

def generate_tag_openai(drug_list:List, model_params:Dict)-> List:
    """Generates tags based on provided prompts and params through OpenAI API call.
    
    Args:
        drug_list: list- list of drugs for which tags should be generated.
        model_params: Dict - parameters dictionary for openAI API call
    Returns
        List of tags generated by the API call.
    """
    tag_list = []
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    for drug in tqdm(drug_list):
        output = client.chat.completions.create(
            model=model_params.get('model'),
            messages=[
                    {"role": "system", "content": model_params.get('prompt')},
                    {"role": "user", "content": drug}
                ],
            temperature= model_params.get('temperature')
        )
        #print (drug)
        tag_list.append(output.choices[0].message.content)
    return tag_list

#########################################
##### NODES HERE ########################
#########################################

def add_most_permissive_marketing_tags_fda(in_list: pd.DataFrame) -> pd.DataFrame:
    cache = {}
    for idx, row in tqdm(in_list.iterrows(), total=len(in_list), desc = "caching drug marketing statuses"):
        if row['Ingredient'] not in cache:
            cache[row['Ingredient']]=getMostPermissiveStatus(getAllStatuses(in_list, row['Ingredient']))
    new_approval_tags_column = []
    for idx, row in tqdm(in_list.iterrows(), total = len(in_list), desc = "adding marketing status labels"):
        new_approval_tags_column.append(cache[row['Ingredient']])
    in_list['Type']=new_approval_tags_column
    in_list.rename(columns={"Type":"marketing_status_usa"}, inplace=True)
    return in_list

def drop_discontinued_drugs(drug_list: pd.DataFrame, column_name: str, discontinued_string: str):
    indices_to_drop = []
    for idx, row in tqdm(drug_list.iterrows(), total=len(drug_list),desc="dropping discontinued drugs"):
        if row[column_name]==discontinued_string:
            indices_to_drop.append(idx)  
    return drug_list.drop(indices_to_drop).drop_duplicates(subset=['drug_name'])

def remove_manually_excluded_drugs(list_in: pd.DataFrame, exclusions: list[str], drug_name_column: str) -> pd.DataFrame:
    """
    Args:
        list_in (pandas.DataFrame): data frame with a list of drugs, containing column "{drug_name_column}"
        exclusions (list[str]): drugs to exclude from list
        drug_name_column (string): what the drug name column is called

    Returns:
        pd.DataFrame: list_in with exclusions removed.
    """

    indices_to_drop = []
    for idx, row in list_in.iterrows():
        if row[drug_name_column] in exclusions:
            indices_to_drop.append(idx)
    return list_in.drop(indices_to_drop)

def create_standardized_columns(df_in: pd.DataFrame, drug_name_column:str, approval_date_column:str) -> pd.DataFrame:
    """
    Args:
        df_in (pd.DataFrame): raw data converted to dataframe
        drug_name_column (str): name of drug column in raw data source
        approval_date_column (str): name of approval date column 
    Returns:
        pd.DataFrame: raw data converted to dataframe with drug name and approval date columns renamed so we can use them later.
    """
    df_in.rename(
        columns={drug_name_column:'drug_name',
                 approval_date_column:'approval_date',
                 },
        inplace=True
        )
    # remove illegal unicode characters from list:
    df_in = remove_illegal_characters_openpyxl(df_in)
    new_names = []
    for idx, row in df_in.iterrows():
        if type(row['drug_name'])==float:
            new_names.append("Error")
        else:
            new_names.append(row['drug_name'].upper())
    df_in['drug_name'] = new_names
    df_in.drop_duplicates(subset='drug_name', inplace=True)
    return df_in

def standardize_approval_date_format(df_in: pd.DataFrame) -> pd.DataFrame:
    for idx, row in tqdm(df_in.iterrows(), total=len(df_in), desc = "standardizing date format"):
        approval_date_string = row['approval_date']
        prompt = f"Please convert the following string: ({approval_date_string}) to YYYYMMDD format. If the date is on or before a certain date, just return the named date"

def tag_combination_therapies(inputList: pd.DataFrame, delimiters: list[str], exclusions: list[str])->pd.DataFrame:
    combination_therapy = []
    for idx, row in tqdm(inputList.iterrows(), total=len(inputList)):
        combination_therapy.append(is_combination_therapy(row['drug_name'], delimiters, exclusions))
    inputList['combination_therapy'] = combination_therapy
    return inputList

def identify_drugs(input_list: pd.DataFrame, params: dict) -> pd.DataFrame:
    ids = []
    labels = []
    ids_cache = {}
    labels_cache = {}
    for idx,row in tqdm(input_list.iterrows(), total=len(input_list), desc="identifying drugs"):
        drug_name = row['drug_name']
        if drug_name in ids_cache:
            ids.append(ids_cache[drug_name])
            labels.append(labels_cache[drug_name])
        else:
            if row['combination_therapy']:
                prefer_rxcui = True
            else:
                prefer_rxcui = False
            item_curie, item_label = identify(drug_name, params, prefer_rxcui)
            ids.append(item_curie)
            labels.append(item_label)
            ids_cache[drug_name]=item_curie
            labels_cache[drug_name]=item_label
    
    input_list['curie']=ids
    input_list['curie_label']=labels
    return input_list

def add_approval_tags(original_dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    args:
        original_dataframe (pd.DataFrame): current drug list data frame
        column_name (str): what to call the resulting column
    returns:
        new_dataframe (pd.DataFrame): drug list with approval tags
    """
    original_dataframe[column_name]="TRUE"
    return original_dataframe

def add_ingredients(input_list: pd.DataFrame, delimiters: list[str]):
    ingredients_list = []
    for idx,row in tqdm(input_list.iterrows(), total=len(input_list)):
        if row['combination_therapy']==True:
            ingredients_list.append(multi_split(row['drug_name'], delimiters))
        else:
            ingredients_list.append(None)

    input_list["ingredients_list"]=ingredients_list
    return input_list

def desalt_drugs(in_list: pd.DataFrame, desalting_params: dict) -> pd.DataFrame:
    new_name_col = []
    for idx, row in tqdm(in_list.iterrows(), total=len(in_list)):
        if type(row["drug_name"])==float:
            new_name_col.append("Error")
        else:
            curr_name_col = []
            if row['combination_therapy']==True:
                new_name_items = []
                for item in string_to_list(row['ingredients_list']):
                    new_name_items.append(removeCationsAnionsAndBasicTerms(item, desalting_params).strip())
                new_name_items.sort()
                new_name_col.append("; ".join(new_name_items))
            else:
                new_name_col.append(removeCationsAnionsAndBasicTerms(row['drug_name'], desalting_params))
    in_list['drug_name']=new_name_col
    return in_list

def add_unlisted_single_ingredients(input_list: pd.DataFrame) -> pd.DataFrame:
    ingList = list(input_list['drug_name'])
    for idx, row in tqdm(input_list.iterrows(), total=len(input_list), desc="adding unlisted single therapies"):
        if row['combination_therapy']:
            for item in string_to_list(row['ingredients_list']):
                if item not in ingList:
                    new_columns = {
                        'drug_name':item.strip(), 
                        'approval_date':row['approval_date'],
                        'combination_therapy':False
                        }
                    input_list = add_row(input_list, new_columns)
                    ingList = list(input_list['drug_name'])
    return input_list

def add_ingredient_ids(input_list: pd.DataFrame, nameres_params) -> pd.DataFrame:
    cache = {}
    ingredient_ids_list=[]
    for idx,row in tqdm(input_list.iterrows(), total=len(input_list), desc="adding ingredient IDs"):
        curr_row_ingredient_ids = []
        if not row['combination_therapy']:
            ingredient_ids_list.append(None)
        else:
            for ingredient in string_to_list(row['ingredients_list']):
                if ingredient in cache:
                    curr_row_ingredient_ids.append(cache[ingredient])
                else:
                    curie,label = identify(ingredient, nameres_params, False)
                    curr_row_ingredient_ids.append(curie)
                    cache[ingredient]=curie
            ingredient_ids_list.append(curr_row_ingredient_ids)   
    input_list["ingredient_ids"] = ingredient_ids_list
    return input_list

def check_nameres_single_entry(input_disease: str, id_label: str, params: dict) -> str:
    """
    Args: 
        inputDisease (str): the name of the disease extracted from indications text using LLMs
        params (dict): LLM parameters

    Returns:
        str: the ID of the disease as interpreted by the LLM, or "NONE"

    """

    prompt = f"{params.get('prompt')} Concept 1: {input_disease}. Concept 2: {id_label}"
    print(prompt)
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    output = client.chat.completions.create(
            model=params.get('model'),
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_disease}
                ],
            temperature= params.get('temperature')
        )
    response = output.choices[0].message.content
    print(response)
    return response

def check_nameres_llm(inList: pd.DataFrame, concept_name_column: str, nameres_label_column: str, params:dict, llm_opinion_column: str):
    tags = []
    cache = {}
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="applying LLM ID check"):
        concept = row[concept_name_column]
        nameres_label= row[nameres_label_column]
        if concept in cache:
            tags.append(cache[concept])
        else:
            try:
                llm_id = check_nameres_single_entry(concept, nameres_label, params.get("model_params"))
                cache[concept] = llm_id
                tags.append(llm_id)
            except:
                tags.append("ERROR")
        
    inList[llm_opinion_column]=tags
    return inList


def clean_bad_entries (inList: pd.DataFrame, column_name: str, error_string: str):
    indices_to_drop = []
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="cleaning bad entries..."):
        if row[column_name] == error_string:
            indices_to_drop.append(idx)
    
    inList.drop(indices_to_drop, inplace=True)

    return inList

def get_curie(string, biolink_type, limit, autocomplete:str):
    itemRequest = f"https://name-resolution-sri.renci.org/lookup?string={string}&autocomplete={autocomplete}&offset=0&limit={limit}&biolink_type={biolink_type}"
    return nameres(itemRequest)

@cache
def nameres(itemRequest:str) -> str:
    returned = (pd.read_json(StringIO(requests.get(itemRequest).text)))
    resolvedCurie = returned.curie
    resolvedLabel = returned.label
    return resolvedCurie, resolvedLabel

def choose_best_id (concept: str, ids: list[str], labels: list[str], params: dict) -> str:
    ids_and_names = []
    for idx, item in enumerate(ids):
        ids_and_names.append(f"{idx+1}: {item} ({labels[idx]})")   
    ids_and_names = ";\n".join(ids_and_names)
    prompt = f"{params.get('prompt')} "
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    #print(f"Drug Concept: {concept}. \r\n\n Options: {ids_and_names}")
    output = client.chat.completions.create(
            model=params.get('model'),
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content":  f"Drug Concept: {concept}. \r\n\n Options: {ids_and_names}"}
                ],
            temperature= params.get('temperature')
        )
    print(output.choices[0].message.content)
    return output.choices[0].message.content

def llm_improve_ids(inList: pd.DataFrame, concept_column_name: str, params: dict, biolink_type: str, first_attempt_column_name: str, llm_decision_column: str, new_best_id_column: str ):
    print(inList)
    print("Improving IDs with LLM best-choice selection")
    new_ids = []
    cache = {}
    for idx, row in tqdm(inList.iterrows(), total=len(inList), desc="Using LLM to choose best of top 30 nameres hits for each flagged entry"):
        
        concept = row[concept_column_name]
        llm_decision = row[llm_decision_column]
        if llm_decision == True or (type(llm_decision)==str and llm_decision.upper()=="TRUE"):
            new_ids.append(row[first_attempt_column_name])
        else:
            if concept in cache:
                new_ids.append(cache[concept])
            else:
                try:
                    ids, labels = get_curie(string=concept, biolink_type=biolink_type, limit=30, autocomplete="false")
                    best_id = choose_best_id(concept, ids, labels, params.get('model_params'))
                    # append and cache best id from LLM
                    new_ids.append(best_id)
                    cache[concept]=best_id
                except Exception as e:
                    print("error during llm id improvement")
                    print(e)

                    new_ids.append("ERROR")  
    inList[new_best_id_column] = new_ids
    return clean_bad_entries(clean_bad_entries(inList, new_best_id_column, "ERROR"), new_best_id_column, "NONE")


def add_alternate_ids(input_list: pd.DataFrame) -> pd.DataFrame:
    cache = {}
    label_cache = {}
    alternate_ids = []
    labels = []
    for idx, row in tqdm(input_list.iterrows(), total=len(input_list), desc="adding alternate IDs from Node Normalizer"):
        curie = row['improved_id']
        if curie in cache:
            alternate_ids.append(cache[curie])
            labels.append(label_cache[curie])
        else:
            item_alternate_ids, label = normalize(curie)
            cache[curie]=item_alternate_ids
            label_cache[curie]=label
            alternate_ids.append(item_alternate_ids)
            labels.append(label)

    input_list['alternate_ids']=alternate_ids
    input_list['label']=labels
    return input_list

def return_final_list(input_list: pd.DataFrame, properties: list[str], approval_tag: str, additional_fields: Optional[list[str]]) -> pd.DataFrame:
    """
    Args:
        input_list (pd.DataFrame): list with lots of features
        properties (list[str]): list of properties we actually want to save into final list.
        approval_tag (str): approval tag column name for this list.
    Returns:
        pd.DataFrame: input list with just properties.
    """
    if additional_fields:
        properties.extend(additional_fields)
    properties.append(approval_tag)
    print(f"generating list with properties: {properties}")
    return input_list[properties]

def reformat_ingredients_with_ollama(in_list: pd.DataFrame, prompt_string) -> pd.DataFrame:
    new_drug_names = []
    for idx, row in tqdm(in_list.iterrows(), total=len(in_list), desc="renaming poorly formatted drug names"):
        prompt = f"{prompt_string}: Drug: {row['drug_name']}"
        #output = query_ollama(prompt)

        output = query_ollama(
            prompt=prompt,
            temperature=0.3,
            max_tokens=100
        )
        print(output['response'])
        new_drug_names.append(output["response"])
    in_list['drug_name'] = new_drug_names
    return in_list

def merge_all_drug_lists(pmda: pd.DataFrame, ema: pd.DataFrame, orangebook: pd.DataFrame, purplebook: pd.DataFrame, russia: pd.DataFrame, india: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple dataframes based on the 'curie' field, combining matching rows.
    
    Parameters:
    df_list (list): List of pandas dataframes to merge
    
    Returns:
    pandas.DataFrame: Merged dataframe with combined rows
    """
    df_list = list([pmda, ema, orangebook, purplebook, russia, india])
    if not df_list:
        raise ValueError("Empty list of dataframes provided")
        
    if len(df_list) == 1:
        return df_list[0]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    def combine_rows(series):
        unique_values = series.dropna().unique()
        if len(unique_values) == 0:
            return np.nan
        elif len(unique_values) == 1:
            return unique_values[0]
        else:
            # If multiple non-null values exist, join them with semicolons
            return '; '.join(str(x) for x in unique_values)
    
    # Group by 'curie' and aggregate all other columns
    merged_df = combined_df.groupby('improved_id', as_index=False).agg(combine_rows)
    return merged_df


def enrich_drug_list(drug_list:List, params:Dict, llm_to_use)-> pd.DataFrame:
    """Node enriching existing drug list with llm-generated tags.
    
    Args:
        drug_list: pd.DataFrame - merged drug_list with drug names column that will be used for tag generation
        params: Dict - parameters dictionary specifying tag names, column names, and model params
        llm_to_use: str - name of LLM to use (could include openAI or local Llama instance via ollama)
    Returns
        pd.DataFrame with x new tag columns (where x corresponds to number of tags specified in params)
    """
    for tag in params.keys():
        print(f"applying tag: \'{tag}\' to drug list")
        input_col = params[tag].get('input_col')
        output_col = params[tag].get('output_col')
        model_params = params[tag].get('model_params')
        if llm_to_use == "openai":
            drug_list[output_col] = generate_tag_openai(drug_list[input_col], model_params)
        if llm_to_use == "localLlama":
            drug_list[output_col] = generate_tag_localLlama(drug_list[input_col], model_params)
        else:
            print("error")
    return drug_list

def get_smiles_from_pubchem(pubchem_id: int) -> Optional[str]:
    """
    Retrieve the SMILES string for a chemical compound using its PubChem ID (CID).
    
    Args:
        pubchem_id (int): The PubChem Compound ID (CID)
    
    Returns:
        Optional[str]: The SMILES string if found, None if not found or error occurs
        
    Raises:
        ValueError: If the pubchem_id is not a positive integer
        requests.RequestException: If there's an error with the API request
    """
    if not isinstance(pubchem_id, int) or pubchem_id <= 0:
        raise ValueError("PubChem ID must be a positive integer")

    # PubChem REST API endpoint
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    endpoint = f"{base_url}/compound/cid/{pubchem_id}/property/IsomericSMILES/JSON"

    try:
        # Make the API request
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the JSON response
        data = response.json()
        
        # Extract the SMILES string
        smiles = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        return smiles
        
    except requests.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
def add_approval_false_tags(list_in:pd.DataFrame, tag_names:list[str]) -> pd.DataFrame:
    for tag_name in tag_names:
        new_tags_col = []
        for idx, row in tqdm(list_in.iterrows(), total=len(list_in), desc=f"adding unapproved tags for tag {tag_name}"):
            if row[tag_name]==1:
                new_tags_col.append(True)
            else:
                new_tags_col.append(False)
        list_in[tag_name]=new_tags_col
    return list_in

def extract_pubchem_id(identifier):
    """
    Extract the numeric ID from PubChem identifier strings.
    
    Args:
        identifier (str): String containing PubChem identifier
            e.g., "PUBCHEM:1234" or "PUBCHEM.COMPOUND:1234" or "1234"
            
    Returns:
        str: The extracted ID or original string if no pattern is matched
    
    Examples:
        >>> extract_pubchem_id("PUBCHEM:1234")
        "1234"
        >>> extract_pubchem_id("PUBCHEM.COMPOUND:1234")
        "1234"
        >>> extract_pubchem_id("1234")
        "1234"
    """
    if "PUBCHEM.COMPOUND:" in identifier:
        return identifier.split("PUBCHEM.COMPOUND:")[-1]
    elif "PUBCHEM:" in identifier:
        return identifier.split("PUBCHEM:")[-1]
    return identifier



def add_SMILES_strings(drug_list: pd.DataFrame) -> pd.DataFrame:
    smiles = []
    for idx, row in tqdm(drug_list.iterrows(), total=len(drug_list)):
        #print(row['curie'])
        identifier = row['curie']
        if "PUBCHEM" in identifier:
            pc_id = int(extract_pubchem_id(identifier))
            #print(pc_id)
            smiles.append(get_smiles_from_pubchem(pc_id))
        else:
            smiles.append("")
    drug_list['smiles']=smiles
    return drug_list

def translate_dataframe(df, source_lang='ru', dest_lang='en'):
    """
    Translate both column names and data content from source language to destination language.
    This is a synchronous wrapper function that calls the async implementation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to translate
    source_lang : str, default='ru'
        Source language code (Russian by default)
    dest_lang : str, default='en'
        Destination language code (English by default)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with translated column names and data
    """
    # Run the async function in an event loop
    return asyncio.run(_translate_dataframe_async(df, source_lang, dest_lang))


async def _translate_dataframe_async(df, source_lang='ru', dest_lang='en'):
    """
    Async implementation of the translation function.
    
    Parameters are the same as translate_dataframe.
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    async with Translator() as translator:
        # First translate column names
        print(f"Translating {len(df.columns)} column names from {source_lang} to {dest_lang}...")
        column_mapping = {}
        
        for column in tqdm(df.columns, desc="Translating columns", unit="column"):
            try:
                result = await translator.translate(column, src=source_lang, dest=dest_lang)
                column_mapping[column] = result.text
            except Exception as e:
                print(f"Error translating column '{column}': {e}")
                column_mapping[column] = column  # Keep original on error
        
        # Rename the columns in the DataFrame
        df_copy = df_copy.rename(columns=column_mapping)
        print("Column translation complete!")
        
        # Now translate the text data in each cell
        print(f"Translating data content from {source_lang} to {dest_lang}...")
        
        # Get only the text columns (skip numeric columns)
        text_columns = df_copy.select_dtypes(include=['object']).columns
        
        if len(text_columns) == 0:
            print("No text columns found to translate.")
            return df_copy
            
        # Calculate total cells to translate for progress bar
        total_cells = sum(df_copy[col].notna().sum() for col in text_columns)
        
        # Create a new dataframe for translated content
        translated_df = df_copy.copy()
        
        with tqdm(total=total_cells, desc="Translating cells", unit="cell") as pbar:
            cache = {}
            for col in text_columns:
                for idx in df_copy.index:
                    value = df_copy.at[idx, col]
                    
                    # Only translate if value is a string and not empty
                    if isinstance(value, str) and value.strip():
                        try:
                            # Add small delay to avoid hitting API rate limits
                            
                            if value in cache:
                                result = cache[value]
                            else:
                                await asyncio.sleep(0.1)
                                result = await translator.translate(value, src=source_lang, dest=dest_lang)
                                cache[value]=result
                            translated_df.at[idx, col] = result.text
                        except Exception as e:
                            print(f"Error translating value '{value}': {e}")
                            # Keep original value on error
                    
                    if isinstance(value, str) or pd.notna(value):
                        pbar.update(1)
        
        print("Data translation complete!")
        return translated_df


# For backward compatibility with Kedro pipeline
def translate_dataframe_columns(df, source_lang='ru', dest_lang='en'):
    """
    This function is maintained for backward compatibility.
    Now it fully translates both columns and data.
    """
    return translate_dataframe(df, source_lang, dest_lang)

########################################################################################
######################### ATC CODES ####################################################
########################################################################################

# import pandas as pd
# import requests
# import ast
# import re
# import time
# from urllib.parse import quote

# def get_atc_codes(df, rate_limit_delay=0.1):
#     """
#     Query external databases to find ATC codes for compounds based on their identifiers
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         DataFrame containing 'improved_id' and 'alternate_ids' columns
#     rate_limit_delay : float
#         Delay between API requests in seconds to avoid hitting rate limits
        
#     Returns:
#     --------
#     pandas.DataFrame
#         Original DataFrame with additional columns for ATC codes
#     """
#     # Parse alternate_ids if they're stored as strings
#     def parse_alternate_ids(alternate_ids):
#         if isinstance(alternate_ids, str):
#             try:
#                 return ast.literal_eval(alternate_ids)
#             except:
#                 return []
#         return alternate_ids if isinstance(alternate_ids, list) else []
    
#     df['alternate_ids_list'] = df['alternate_ids'].apply(parse_alternate_ids)
    
#     # Function to extract identifiers by source
#     def extract_ids_by_source(alt_ids):
#         id_dict = {}
#         for id_str in alt_ids:
#             if ':' in id_str:
#                 source, id_val = id_str.split(':', 1)
#                 if source not in id_dict:
#                     id_dict[source] = []
#                 id_dict[source].append(id_val)
#         return id_dict
    
#     df['ids_by_source'] = df['alternate_ids_list'].apply(extract_ids_by_source)
    
#     # Initialize columns for ATC codes
#     df['atc_codes'] = None
    
#     # Process each row to find ATC codes
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="obtaining atc codes"):
#         atc_codes = []
#         ids_by_source = row['ids_by_source']
        
#         # Try PubChem
#         if not atc_codes and 'PUBCHEM.COMPOUND' in ids_by_source:
#             for pc_id in ids_by_source['PUBCHEM.COMPOUND']:
#                 try:
#                     # PubChem API to get ATC codes
#                     response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pc_id}/classification/JSON")
#                     if response.status_code == 200:
#                         data = response.json()
#                         # Parse PubChem response for ATC codes (example logic, adjust as needed)
#                         if 'Hierarchy' in data:
#                             for hierarchy in data['Hierarchy']:
#                                 if 'ATC' in hierarchy.get('SourceName', ''):
#                                     atc_codes.append(hierarchy.get('SourceID', ''))
#                     time.sleep(rate_limit_delay)
#                 except Exception as e:
#                     print(f"Error querying PubChem for {pc_id}: {e}")

# # Try DrugCentral
#         if not atc_codes and 'DrugCentral' in ids_by_source:
#             for dc_id in ids_by_source['DrugCentral']:
#                 try:
#                     response = requests.get(f"https://drugcentral.org/api/drug/{dc_id}")
#                     if response.status_code == 200:
#                         data = response.json()
#                         # Extract ATC codes from DrugCentral response
#                         if 'atc_code' in data:
#                             atc_codes.extend(data['atc_code'])
#                     time.sleep(rate_limit_delay)
#                 except Exception as e:
#                     print(f"Error querying DrugCentral for {dc_id}: {e}")
        
#         # Try ChEMBL
#         if not atc_codes and 'CHEMBL.COMPOUND' in ids_by_source:
#             for chembl_id in ids_by_source['CHEMBL.COMPOUND']:
#                 try:
#                     response = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json")
#                     if response.status_code == 200:
#                         data = response.json()
#                         # Extract ATC codes from ChEMBL response
#                         if 'atc_classifications' in data:
#                             atc_codes.extend(data['atc_classifications'])
#                     time.sleep(rate_limit_delay)
#                 except Exception as e:
#                     print(f"Error querying ChEMBL for {chembl_id}: {e}")
                    
#         # Try KEGG (using CAS as the identifier)
#         if not atc_codes and 'CAS' in ids_by_source:
#             for cas_id in ids_by_source['CAS']:
#                 try:
#                     response = requests.get(f"https://rest.kegg.jp/find/drug/{cas_id}")
#                     if response.status_code == 200:
#                         kegg_drugs = [line.split("\t")[0].replace("dr:", "") for line in response.text.strip().split("\n") if line]
                        
#                         for kegg_drug in kegg_drugs:
#                             drug_response = requests.get(f"https://rest.kegg.jp/get/{kegg_drug}")
#                             if drug_response.status_code == 200:
#                                 # Parse KEGG response for ATC codes
#                                 atc_match = re.search(r'ATC code: ([A-Z]\d{2}[A-Z]{2}\d{2})', drug_response.text)
#                                 if atc_match:
#                                     atc_codes.append(atc_match.group(1))
#                             time.sleep(rate_limit_delay)
#                 except Exception as e:
#                     print(f"Error querying KEGG for {cas_id}: {e}")
        
        
#         # Try DrugBank last (most annoying)
#         if 'DRUGBANK' in ids_by_source:
#             for db_id in ids_by_source['DRUGBANK']:
#                 try:
#                     # DrugBank requires authentication for their API, so we'll use alternative methods
#                     # This example uses a public endpoint that doesn't require authentication
#                     response = requests.get(f"https://go.drugbank.com/drugs/{db_id}.json")
#                     if response.status_code == 200:
#                         data = response.json()
#                         if 'atc_codes' in data:
#                             atc_codes.extend(data['atc_codes'])
#                     time.sleep(rate_limit_delay)  # Respect rate limits
#                 except Exception as e:
#                     print(f"Error querying DrugBank for {db_id}: {e}")
        

        
        
#         # Update the DataFrame with found ATC codes
#         df.at[idx, 'atc_codes'] = atc_codes if atc_codes else None
    
#     # Function to break down ATC code into levels
#     def break_down_atc(atc_code):
#         if not atc_code:
#             return None, None, None, None, None
        
#         # Level 1: Anatomical main group (first character)
#         level1 = atc_code[0] if len(atc_code) >= 1 else None
        
#         # Level 2: Therapeutic subgroup (first 3 characters)
#         level2 = atc_code[:3] if len(atc_code) >= 3 else None
        
#         # Level 3: Pharmacological subgroup (first 4 characters)
#         level3 = atc_code[:4] if len(atc_code) >= 4 else None
        
#         # Level 4: Chemical subgroup (first 5 characters)
#         level4 = atc_code[:5] if len(atc_code) >= 5 else None
        
#         # Level 5: Chemical substance (all 7 characters)
#         level5 = atc_code if len(atc_code) == 7 else None
        
#         return level1, level2, level3, level4, level5
    
#     # Apply the top-level ATC code function
#     def get_top_atc(atc_codes):
#         return atc_codes[0] if isinstance(atc_codes, list) and atc_codes else None
    
#     df['atc_main'] = df['atc_codes'].apply(get_top_atc)
    
#     # Break down the main ATC code into levels
#     df[['atc_level1', 'atc_level2', 'atc_level3', 'atc_level4', 'atc_level5']] = pd.DataFrame(
#         df['atc_main'].apply(break_down_atc).tolist(), 
#         index=df.index
#     )
    
#     return df

###############################################################################
## PRIMARY ATC  (FASTER) ######################################################
###############################################################################



# def get_atc_from_rxnorm(rxnorm_id):
#     """Get ATC code from RxNorm ID using the RxNav API"""
#     try:
#         # First, validate if the input is a valid RxNorm ID (numeric)
#         if not rxnorm_id.isdigit():
#             return None
            
#         # Call the RxNav API to get ATC codes
#         response = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxnorm_id}/property?propName=ATC")
        
#         if response.status_code == 200:
#             data = response.json()
#             prop_concept_group = data.get('propConceptGroup', {})
#             if prop_concept_group:
#                 prop_concepts = prop_concept_group.get('propConcept', [])
#                 atc_codes = [prop['propValue'] for prop in prop_concepts if prop.get('propName') == 'ATC']
#                 return atc_codes if atc_codes else None
#         return None
#     except Exception as e:
#         print(f"Error getting ATC from RxNorm for {rxnorm_id}: {str(e)}")
#         return None

# def get_atc_from_chembl(chembl_id):
#     """Get ATC code from ChEMBL ID using the ChEMBL API"""
#     try:
#         # Remove the 'CHEMBL.COMPOUND:' prefix if present
#         if 'CHEMBL.COMPOUND:' in chembl_id:
#             chembl_id = chembl_id.split(':')[1]
        
#         # Call the ChEMBL API
#         response = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json")
        
#         if response.status_code == 200:
#             data = response.json()
#             atc_classifications = data.get('atc_classifications', [])
#             return atc_classifications if atc_classifications else None
#         return None
#     except Exception as e:
#         print(f"Error getting ATC from ChEMBL for {chembl_id}: {str(e)}")
#         return None

# def get_atc_from_pubchem(pubchem_id):
#     """Get ATC code from PubChem ID using PubChem PUG REST API"""
#     try:
#         # Remove the 'PUBCHEM.COMPOUND:' prefix if present
#         if 'PUBCHEM.COMPOUND:' in pubchem_id:
#             pubchem_id = pubchem_id.split(':')[1]
            
#         # The PubChem API doesn't directly provide ATC codes, so we'll use the classification browser
#         response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{pubchem_id}/JSON")
        
#         if response.status_code == 200:
#             data = response.json()
#             sections = data.get('Record', {}).get('Section', [])
            
#             # Look for ATC codes in the classifications
#             for section in sections:
#                 if section.get('TOCHeading') == 'Classification':
#                     subsections = section.get('Section', [])
#                     for subsection in subsections:
#                         if 'ATC' in subsection.get('TOCHeading', ''):
#                             information = subsection.get('Information', [])
#                             for info in information:
#                                 value = info.get('Value', {}).get('StringWithMarkup', [])
#                                 atc_codes = []
#                                 for val in value:
#                                     atc_match = re.search(r'[A-Z]\d\d[A-Z][A-Z]\d\d', val.get('String', ''))
#                                     if atc_match:
#                                         atc_codes.append(atc_match.group(0))
#                                 if atc_codes:
#                                     return atc_codes
#         return None
#     except Exception as e:
#         print(f"Error getting ATC from PubChem for {pubchem_id}: {str(e)}")
#         return None

# def get_atc_from_drugcentral(drugcentral_id):
#     """Get ATC code from DrugCentral ID using the DrugCentral API"""
#     try:
#         # Extract the numeric ID if it has a prefix
#         if 'DrugCentral:' in drugcentral_id:
#             drugcentral_id = drugcentral_id.split(':')[1]
            
#         # Call the DrugCentral API
#         response = requests.get(f"https://drugcentral.org/api/drugcentral/structures?q={drugcentral_id}")
        
#         if response.status_code == 200:
#             data = response.json()
#             if data and isinstance(data, list) and len(data) > 0:
#                 drug_data = data[0]
#                 atc_codes = []
#                 for annotation in drug_data.get('annotations', []):
#                     if annotation.get('type') == 'ATC':
#                         atc_codes.append(annotation.get('value'))
#                 return atc_codes if atc_codes else None
#         return None
#     except Exception as e:
#         print(f"Error getting ATC from DrugCentral for {drugcentral_id}: {str(e)}")
#         return None

# def get_atc_from_whocc(drug_name):
#     """Get ATC code from WHO Collaborating Centre for Drug Statistics Methodology"""
#     try:
#         # Encode the drug name for URL
#         encoded_name = quote(drug_name)
        
#         # Search the WHO ATC database
#         response = requests.get(f"https://www.whocc.no/atc_ddd_index/?name={encoded_name}")
        
#         if response.status_code == 200:
#             # Parse HTML response to extract ATC codes
#             # This is a placeholder as proper HTML parsing would be needed
#             atc_matches = re.findall(r'[A-Z]\d\d[A-Z][A-Z]\d\d', response.text)
#             return atc_matches if atc_matches else None
#         return None
#     except Exception as e:
#         print(f"Error getting ATC from WHO for {drug_name}: {str(e)}")
#         return None

# def extract_id_from_curie(id_string, prefix):
#     """Extract specific ID from a CURIE string"""
#     for item in id_string.split(','):
#         item = item.strip()
#         if item.startswith(prefix):
#             return item
#     return None

# def get_chebi_drugcentral_xrefs(chebi_id):
#     """
#     Get DrugCentral XREFs from a CHEBI ID using the CHEBI API or OLS
    
#     Parameters:
#     chebi_id (str): CHEBI ID (format: CHEBI:XXXXX)
    
#     Returns:
#     list: List of DrugCentral IDs referenced by this CHEBI entry
#     """
#     try:
#         # Extract the numeric part if needed
#         if ':' in chebi_id:
#             chebi_num = chebi_id.split(':')[1]
#         else:
#             chebi_num = chebi_id
            
#         # Use the EBI OLS API to get cross-references
#         response = requests.get(f"https://www.ebi.ac.uk/ols/api/ontologies/chebi/terms/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FCHEBI_{chebi_num}")
        
#         if response.status_code == 200:
#             data = response.json()
#             xrefs = data.get('annotation', {}).get('database_cross_reference', [])
            
#             # Filter for DrugCentral references
#             drugcentral_ids = []
#             for xref in xrefs:
#                 if isinstance(xref, str) and xref.startswith('DrugCentral:'):
#                     drugcentral_ids.append(xref)
            
#             return drugcentral_ids
#         return []
#     except Exception as e:
#         print(f"Error getting DrugCentral XREFs from CHEBI for {chebi_id}: {str(e)}")
#         return []

# def get_atc_for_row(row):
#     """Process a single row to find ATC code"""
#     # Convert string representation of list to actual list if needed
#     if isinstance(row['alternate_ids'], str):
#         try:
#             alt_ids = ast.literal_eval(row['alternate_ids'])
#         except (SyntaxError, ValueError):
#             alt_ids = [id.strip() for id in row['alternate_ids'].strip('[]').split(',')]
#     else:
#         alt_ids = row['alternate_ids']
    
#     # Add the curie to the list of IDs if not already there
#     if row['improved_id'] not in alt_ids:
#         alt_ids.append(row['curie'])
    
#     # Try each source in order of reliability/accessibility
#     for id_item in alt_ids:
#         id_item = id_item.strip("'\" ")
        
#         # Try ChEMBL
#         if 'CHEMBL.COMPOUND:' in id_item:
#             atc = get_atc_from_chembl(id_item)
#             if atc:
#                 return atc
        
#         # Try PubChem
#         if 'PUBCHEM.COMPOUND:' in id_item:
#             atc = get_atc_from_pubchem(id_item)
#             if atc:
#                 return atc
        
#         # Try DrugCentral
#         if 'DrugCentral:' in id_item:
#             atc = get_atc_from_drugcentral(id_item)
#             if atc:
#                 return atc
    
#     # If no ATC found yet, check if there are CHEBI IDs and try to get DrugCentral XREFs
#     chebi_ids = [id_item.strip("'\" ") for id_item in alt_ids if 'CHEBI:' in id_item]
#     for chebi_id in chebi_ids:
#         # Get DrugCentral XREFs from CHEBI
#         drugcentral_refs = get_chebi_drugcentral_xrefs(chebi_id)
        
#         # Try to get ATC codes from each DrugCentral reference
#         for dc_ref in drugcentral_refs:
#             atc = get_atc_from_drugcentral(dc_ref)
#             if atc:
#                 return atc
    
#     # If we have a name column, try WHO CC as last resort
#     if 'name' in row and row['name']:
#         atc = get_atc_from_whocc(row['name'])
#         if atc:
#             return atc
    
#     return None

# def get_atc_codes_for_dataframe(df, max_workers=5):
#     """
#     Get ATC classifications for all drugs in a dataframe
    
#     Parameters:
#     df (pandas.DataFrame): DataFrame with 'curie' and 'alternate_ids' columns
#     max_workers (int): Maximum number of parallel workers for API calls
    
#     Returns:
#     pandas.DataFrame: Original dataframe with new 'atc_codes' column
#     """
#     # Make a copy to avoid modifying the original
#     result_df = df.copy()
    
#     # Process rows in parallel for efficiency
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Submit all jobs
#         future_to_index = {executor.submit(get_atc_for_row, row): i 
#                           for i, row in df.iterrows()}
        
#         # Collect results
#         atc_codes = [None] * len(df)
#         for future in tqdm(future_to_index):
#             index = future_to_index[future]
#             try:
#                 atc_codes[index] = future.result()
#             except Exception as e:
#                 print(f"Error processing row {index}: {str(e)}")
#                 atc_codes[index] = None
    
#     # Add results to dataframe
#     result_df['atc_codes'] = atc_codes
    
#     return result_df


import pandas as pd
import requests
import ast
import re
import time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # For progress tracking (optional)

def get_atc_codes_from_external_sources(df, num_workers=16, rate_limit_delay=1.0):
    """
    Query external databases in parallel to find ATC codes for compounds based on their identifiers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'improved_id' and 'alternate_ids' columns
    num_workers : int
        Number of parallel workers
    rate_limit_delay : float
        Delay between API requests in seconds to avoid hitting rate limits
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional columns for ATC codes
    """
    # Parse alternate_ids if they're stored as strings
    def parse_alternate_ids(alternate_ids):
        if isinstance(alternate_ids, str):
            try:
                return ast.literal_eval(alternate_ids)
            except:
                return []
        return alternate_ids if isinstance(alternate_ids, list) else []
    
    df['alternate_ids_list'] = df['alternate_ids'].apply(parse_alternate_ids)
    
    # Function to extract identifiers by source
    def extract_ids_by_source(alt_ids):
        id_dict = {}
        for id_str in alt_ids:
            if ':' in id_str:
                source, id_val = id_str.split(':', 1)
                if source not in id_dict:
                    id_dict[source] = []
                id_dict[source].append(id_val)
        return id_dict
    
    df['ids_by_source'] = df['alternate_ids_list'].apply(extract_ids_by_source)
    
    # Function to process a single row
    def process_row(row_data):
        idx, row = row_data
        atc_codes = []
        ids_by_source = row['ids_by_source']
        
        # Try DrugBank first (most reliable for ATC codes)
        if 'DRUGBANK' in ids_by_source:
            for db_id in ids_by_source['DRUGBANK']:
                try:
                    response = requests.get(f"https://go.drugbank.com/drugs/{db_id}.json")
                    if response.status_code == 200:
                        data = response.json()
                        if 'atc_codes' in data:
                            atc_codes.extend(data['atc_codes'])
                    time.sleep(rate_limit_delay)  # Respect rate limits
                except Exception as e:
                    print(f"Error querying DrugBank for {db_id}: {e}")
        
        # Try PubChem
        if not atc_codes and 'PUBCHEM.COMPOUND' in ids_by_source:
            for pc_id in ids_by_source['PUBCHEM.COMPOUND']:
                try:
                    response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pc_id}/classification/JSON")
                    if response.status_code == 200:
                        data = response.json()
                        if 'Hierarchy' in data:
                            for hierarchy in data['Hierarchy']:
                                if 'ATC' in hierarchy.get('SourceName', ''):
                                    atc_codes.append(hierarchy.get('SourceID', ''))
                    time.sleep(rate_limit_delay)
                except Exception as e:
                    print(f"Error querying PubChem for {pc_id}: {e}")
        
        # Try DrugCentral
        if not atc_codes and 'DrugCentral' in ids_by_source:
            for dc_id in ids_by_source['DrugCentral']:
                try:
                    response = requests.get(f"https://drugcentral.org/api/drug/{dc_id}")
                    if response.status_code == 200:
                        data = response.json()
                        if 'atc_code' in data:
                            atc_codes.extend(data['atc_code'])
                    time.sleep(rate_limit_delay)
                except Exception as e:
                    print(f"Error querying DrugCentral for {dc_id}: {e}")
        
        # Try ChEMBL
        if not atc_codes and 'CHEMBL.COMPOUND' in ids_by_source:
            for chembl_id in ids_by_source['CHEMBL.COMPOUND']:
                try:
                    response = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json")
                    if response.status_code == 200:
                        data = response.json()
                        if 'atc_classifications' in data:
                            atc_codes.extend(data['atc_classifications'])
                    time.sleep(rate_limit_delay)
                except Exception as e:
                    print(f"Error querying ChEMBL for {chembl_id}: {e}")
                    
        # Try KEGG (using CAS as the identifier)
        if not atc_codes and 'CAS' in ids_by_source:
            for cas_id in ids_by_source['CAS']:
                try:
                    response = requests.get(f"https://rest.kegg.jp/find/drug/{cas_id}")
                    if response.status_code == 200:
                        kegg_drugs = [line.split("\t")[0].replace("dr:", "") for line in response.text.strip().split("\n") if line]
                        
                        for kegg_drug in kegg_drugs:
                            drug_response = requests.get(f"https://rest.kegg.jp/get/{kegg_drug}")
                            if drug_response.status_code == 200:
                                atc_match = re.search(r'ATC code: ([A-Z]\d{2}[A-Z]{2}\d{2})', drug_response.text)
                                if atc_match:
                                    atc_codes.append(atc_match.group(1))
                            time.sleep(rate_limit_delay)
                except Exception as e:
                    print(f"Error querying KEGG for {cas_id}: {e}")
        
        # Return the row index and the found ATC codes
        return idx, atc_codes
    
    # Process rows in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a list of tasks
        future_to_idx = {executor.submit(process_row, (idx, row)): idx 
                         for idx, row in df.iterrows()}
        
        # Process results as they complete (with optional progress bar)
        for future in tqdm(future_to_idx, total=len(df), desc="Processing compounds"):
            idx, atc_codes = future.result()
            results[idx] = atc_codes
    
    # Update the DataFrame with results
    df['atc_codes'] = pd.Series(results)
    
    # Function to break down ATC code into levels
    def break_down_atc(atc_code):
        if not atc_code:
            return None, None, None, None, None
        
        # Level 1: Anatomical main group (first character)
        level1 = atc_code[0] if len(atc_code) >= 1 else None
        
        # Level 2: Therapeutic subgroup (first 3 characters)
        level2 = atc_code[:3] if len(atc_code) >= 3 else None
        
        # Level 3: Pharmacological subgroup (first 4 characters)
        level3 = atc_code[:4] if len(atc_code) >= 4 else None
        
        # Level 4: Chemical subgroup (first 5 characters)
        level4 = atc_code[:5] if len(atc_code) >= 5 else None
        
        # Level 5: Chemical substance (all 7 characters)
        level5 = atc_code if len(atc_code) == 7 else None
        
        return level1, level2, level3, level4, level5
    
    # Apply the top-level ATC code function
    def get_top_atc(atc_codes):
        return atc_codes[0] if isinstance(atc_codes, list) and atc_codes else None
    
    df['atc_main'] = df['atc_codes'].apply(get_top_atc)
    
    # Break down the main ATC code into levels
    df[['atc_level1', 'atc_level2', 'atc_level3', 'atc_level4', 'atc_level5']] = pd.DataFrame(
        df['atc_main'].apply(break_down_atc).tolist(), 
        index=df.index
    )
    
    return df

# def get_atc(instring:str) -> str:
#     types = ['CHEBI', '']

# def get_xrefs(CHEBI_ID:str) -> list[str]:
#     # code to seek CHEBI get request json
#     # code to find section with xrefs
#     # code to list and return xrefs by type

# def find_atc(CHEBI_ID:str) -> str:
#     atc = None
#     atc = get_atc(CHEBI_ID)
#     if atc:
#         return atc
#     if not atc:
#         xrefs = get_xrefs(CHEBI_ID)
#         for xref in xrefs:
#             atc = get_atc(xref)
#             if atc
#             return atc
    
#     return None

########################################################################################
##################### END ATC CODES ####################################################
########################################################################################



def remap_columns(df_in:pd.DataFrame, colname_in, colname_out) -> pd.DataFrame:
    df_in[colname_out] = df_in[colname_in]

    return df_in


# def obtain_normalized_labels(df_in: pd.DataFrame, id_col_name: str, norm_id_col: str,  norm_label_col: str) -> pd.DataFrame:
#     ids = []
#     labels = []
#     for idx, row in tqdm(df_in.iterrows(), total=len(df_in), desc="normalizing final IDs"):
#         id_in = row[id_col_name]



##########################################################################
############# BATCH TAG ##################################################
##########################################################################
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI



from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_features(input_df: pd.DataFrame, new_feature_name: str, feature_description: str):
    """
    Generate new features for a pandas DataFrame using GPT-4o-mini without batching.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the text to analyze
    text_column : str
        The name of the column containing text to analyze
    new_feature_names : list
        List of names for the new feature columns to be created
    feature_descriptions : list
        List of descriptions for what each feature should represent
    
    Returns:
    --------
    pandas.DataFrame
        The original dataframe with new feature columns added
    """
    
    # Create a copy of the input DataFrame to avoid the SettingWithCopyWarning
    df = input_df.copy()
   
    # Load the API key from .env file
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize the OpenAI client
    model = ChatOpenAI(name="gpt-4o-mini", max_retries=3, model_kwargs={"response_format": {"type": "json_object"}})

    # Create empty columns for the new features
    df[new_feature_name] = None 

    # Create a system prompt for the for the new features
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            f"""You are a medical doctor trying to categorize a list of drugs. 
            For each drug name, extract the following features: 
            "{new_feature_name}: {feature_description}".
            Return ONLY a JSON object with the feature names as keys and the extracted values. 
            No explanations or other text."""
        ),
        (
            "user", 
            "drug to analyze: {drug_name}"
        )
    ])
    chain = prompt | model
    response = chain.batch(list(df['label']), config={"max_concurrency": 50})

    feature_df = pd.DataFrame([json.loads(r.content) for r in response])
    df.update(feature_df)

    return df


# def generate_features(input_df: pd.DataFrame, new_feature_name: str, feature_description: str):
    
#     """
#     Generate new features for a pandas DataFrame using GPT-4o-mini without batching.
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         The dataframe containing the text to analyze
#     text_column : str
#         The name of the column containing text to analyze
#     new_feature_names : list
#         List of names for the new feature columns to be created
#     feature_descriptions : list
#         List of descriptions for what each feature should represent
    
#     Returns:
#     --------
#     pandas.DataFrame
#         The original dataframe with new feature columns added
#     """
    
#     # Create a copy of the input DataFrame to avoid the SettingWithCopyWarning
#     df = input_df.copy()
   
#     # Load the API key from .env file
#     api_key = os.getenv("OPENAI_API_KEY")

#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable not set")
    
#     # Initialize the OpenAI client
#     #'max_retries':8
#     model = ChatOpenAI(name="gpt-4o-mini", model_kwargs={"response_format": {"type": "json_object"}})

#     # Create empty columns for the new features
#     df[new_feature_name] = None 

#     # Create a system prompt for the for the new features
#     prompt = ChatPromptTemplate.from_messages([
#         (
#             "system", 
#             f"""You are a medical doctor trying to categorize a list of drugs. 
#             For each drug name, extract the following features: 
#             "{new_feature_name}: {feature_description}".
#             Return ONLY a JSON object with the feature names as keys and the extracted values. 
#             No explanations or other text."""
#         ),
#         (
#             "user", 
#             "Drug to analyze: {drug_name}"
#         )
#     ])


#     chain = prompt | model
#     #"max_concurrency":50
#     response = chain.batch(list(df['label']), config={"max_concurrency":50})

#     feature_df = pd.DataFrame([json.loads(r.content) for r in response])
#     df.update(feature_df)

#     return df

def extract_outputs_and_prompts(data_dict):
    output_cols = []
    prompts = []
    
    # Extract output_col and prompt for each tag
    for tag_name, tag_info in data_dict.items():
        output_cols.append(tag_info['output_col'])
        prompts.append(tag_info['model_params']['prompt'])
    
    return output_cols, prompts


def add_tags(in_df: pd.DataFrame, tags:dict ) -> pd.DataFrame:
    df = in_df.copy()
    feature_names, feature_descriptions = extract_outputs_and_prompts(tags)
    for feature_name, feature_description in zip(feature_names, feature_descriptions):
        if feature_name not in df.columns:
            df = generate_features(df, feature_name, feature_description)
            print(f"{feature_name} generated")
        else:
            print(f"{feature_name} already exists")
    print(feature_names)
    print(feature_descriptions)

    return df


def filter_drugs(in_df:pd.DataFrame) -> pd.DataFrame:
    indices_to_remove = []
    for idx, row in tqdm(in_df.iterrows(), total=len(in_df), desc="clearing allergens, radioisotopes, and drugs of low therapeutic value"):
        if row['is_allergen'] or row['is_radioisotope_or_diagnostic_agent'] or row['is_no_therapeutic_value']:
            indices_to_remove.append(idx)
    in_df.drop(indices_to_remove, axis=0, inplace=True)
    return in_df