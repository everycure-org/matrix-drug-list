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
    model: str = "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:IQ2_S",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
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
            alternate_ids = output[item]['equivalent_identifiers']
            returned_ids = list(item['identifier'] for item in alternate_ids)
            success = True
        except:
            #print('name resolver error')
            failedCounts += 1
        if failedCounts >= 5:
            return "Error"
    return returned_ids 

def generate_tag_localLlama(drug_list:List, model_params:Dict) -> List:
    tag_list = []
    for drug in tqdm(drug_list):
        output = query_ollama(f"{model_params.get('prompt')} drug: {drug}")
        tag_list.append(output["response"])
    return tag_list
    # Example usage:
    # if __name__ == "__main__":
    #     try:
    #         # Basic usage
    #         response = query_ollama("What is the capital of France?")
    #         print("Basic Response:", response["response"])
            
    #         # Advanced usage with parameters
    #         response = query_ollama(
    #             prompt="Write a short poem about AI",
    #             system_prompt="You are a creative poetry assistant",
    #             temperature=0.9,
    #             max_tokens=100
    #         )
    #         print("\nAdvanced Response:", response["response"])
            
    #     except Exception as e:
    #         print(f"Error: {str(e)}")

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
    output = client.chat.completions.create(
            model=params.get('model'),
            messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content":  f"Drug Concept: {concept}. \r\n\n Options: {ids_and_names}"}
                ],
            temperature= params.get('temperature')
        )
    return output.choices[0].message.content

def llm_improve_ids(inList: pd.DataFrame, concept_column_name: str, params: dict, biolink_type: str, first_attempt_column_name: str, llm_decision_column: str, new_best_id_column: str ):
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
                    print(e)
                    new_ids.append("ERROR")  
    inList[new_best_id_column] = new_ids
    return clean_bad_entries(clean_bad_entries(inList, new_best_id_column, "ERROR"), new_best_id_column, "NONE")



def add_alternate_ids(input_list: pd.DataFrame) -> pd.DataFrame:
    cache = {}
    alternate_ids = []
    for idx, row in tqdm(input_list.iterrows(), total=len(input_list), desc="adding alternate IDs from Node Normalizer"):
        curie = row['curie']
        if curie in cache:
            alternate_ids.append(cache[curie])
        else:
            item_alternate_ids = normalize(curie)
            cache[curie]=item_alternate_ids
            alternate_ids.append(item_alternate_ids)

    input_list['alternate_ids']=alternate_ids
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

def merge_all_drug_lists(pmda: pd.DataFrame, ema: pd.DataFrame, orangebook: pd.DataFrame, purplebook: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple dataframes based on the 'curie' field, combining matching rows.
    
    Parameters:
    df_list (list): List of pandas dataframes to merge
    
    Returns:
    pandas.DataFrame: Merged dataframe with combined rows
    """
    df_list = list([pmda, ema, orangebook, purplebook])
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
    merged_df = combined_df.groupby('curie', as_index=False).agg(combine_rows)
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
    
def add_approval_false_tags(list_in, tag_name) -> pd.DataFrame:
    new_tags_col = []
    for idx, row in tqdm(list_in.iterrows(), total=len(list_in), desc=f"adding unapproved tags for tag{tag_name}"):
        if row[tag_name]==1:
            new_tags_col.append("True")
        else:
            new_tags_col.append("False")
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




# def translate_dataframe_columns(df, source_lang='ru', dest_lang='en'):
#     """
#     Translate all column names from source language to destination language.
#     This is a synchronous wrapper function that calls the async implementation.
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         DataFrame with column names to translate
#     source_lang : str, default='ru'
#         Source language code (Russian by default)
#     dest_lang : str, default='en'
#         Destination language code (English by default)
    
#     Returns:
#     --------
#     pandas.DataFrame
#         DataFrame with translated column names
#     """
#     # Run the async function in an event loop
#     return asyncio.run(_translate_dataframe_columns_async(df, source_lang, dest_lang))


# async def _translate_dataframe_columns_async(df, source_lang='ru', dest_lang='en'):
#     """
#     Async implementation of the translation function.
    
#     Parameters are the same as translate_dataframe_columns.
#     """
#     print(f"Translating {len(df.columns)} column names from {source_lang} to {dest_lang}...")
    
#     async with Translator() as translator:
#         # Create a dictionary to store the translations
#         column_mapping = {}
        
#         # Use tqdm for progress tracking
#         for column in tqdm(df.columns, desc="Translating columns", unit="column"):
#             result = await translator.translate(column, src=source_lang, dest=dest_lang)
#             column_mapping[column] = result.text
        
#         # Rename the columns in the DataFrame
#         df_translated = df.rename(columns=column_mapping)
        
#         print("Translation complete!")
#         return df_translated
    

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