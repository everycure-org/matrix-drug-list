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
from typing import List, Dict, Optional
from openai import OpenAI

testing = False
limit = 1000 # limit for testing full pipeline with limited number of names per list

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

def Normalize(item: str):
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

def get_equivalent_ids(input_list: list[str]) -> list[str]:
    normalized_IDs = []
    for item in tqdm(input_list):
        normalized_IDs.append(Normalize(item))
    return normalized_IDs
        

def getCurie(name, params):
    """
    Args:
        name (str): string to be identified
        params (tuple): name resolver parameters to feed into get request
    
    Returns:
        resolvedName (list[str]): IDs most closely matching string.
        resolvedLabel (list[str]): List of labels associated with respective resolvedName.

    """
    #return [name], [name] #only for testing
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
            return "Error", "Error"
    return resolvedName, resolvedLabel


def getCombinationTherapiesAndSingleTherapiesLists(orangebook: pd.DataFrame, exclusions):
    """
    Args:
        orangebook: pandas.DataFrame
        exclusions: pandas.DataFrame
    
    Returns:
        list: combination therapies
        list: single therapies

    """
    obCombinationTherapies = []
    obSingleTherapies = []
    ingredientList = set(list(orangebook.Ingredient))
    for item in ingredientList:
        if (";" in item) or (" AND " in item) or ("W/" in item):
            obCombinationTherapies.append(item)
        else:
            obSingleTherapies.append(item.strip())
    return list(set(obCombinationTherapies)), list(set(obSingleTherapies))


def isCombinationTherapy(item: str, exclusions: list[str]) -> bool:
    if ((";" in item) or (" AND " in item) or ("W/" in item)) and item not in exclusions:
        return True
    return False


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


def split_therapy_fda(combination_therapy_name):
    """
    Args: 
        combination_therapy_name (str): full combination therapy string including delimiters.
    
    Returns:
        list[str]: a list with the delimiters and whitespace stripped.
    
    """
    ingList = re.split('; | ; | AND | W/ ', combination_therapy_name)
    items_list = list(set(ingList))
    items_list.sort()
    return [x.strip() for x in items_list]


def getIngredientCuries(items_list, name_resolver_params):
    """
    Args:
        items_list(list[str]): list of active ingredients in the therapy.
        name_resolver_params(dict): parameters fed into name resolve to acquire IDs
    Returns:
        list[str]: list containing the best ID for each ingredient in the therapy.
    
    """
    ingredientCuriesList = []
    for i in items_list:
        curie, label = getCurie(i, name_resolver_params)
        ingredientCuriesList.append(curie[0])

    return ingredientCuriesList 


def add_approval_tags(original_dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    args:
        original_dataframe (pd.DataFrame): current drug list data frame
        column_name (str): what to call the resulting column
    returns:
        new_dataframe (pd.DataFrame): drug list with approval tags
    """
    original_dataframe[column_name]=True
    return original_dataframe

def generate_ob_df(drugList: list[str], desalting_params, name_resolver_params, rawdata, split_exclusions, approval_tags_name ) -> pd.DataFrame:
    
    Approved_USA, combination_therapy, therapyName, name_in_orange_book, available_USA, curie_ID, curie_label, ingredient_curies = ([] for i in range(8))
    for index, item in tqdm(enumerate(list(drugList)), total=len(drugList)):
        if not testing or testing and index < limit:
            originalItem = item
            # Things that get updated in the same way whether the therapy is a combination therapy or not 
            name_in_orange_book.append(originalItem) #1
            #Approved_USA.append("True") #2
            available_USA.append(getMostPermissiveStatus(getAllStatuses(rawdata,item)))#3

            if isCombinationTherapy(item, split_exclusions): # Combination Therapy Handling
                combination_therapy.append("True")#4
                items_list = split_therapy_fda(originalItem)
                new_therapies = list(i for i in items_list if i not in drugList)
                for i in new_therapies:
                    drugList.add(i)
                newIngList = list(removeCationsAnionsAndBasicTerms(i.strip(), desalting_params).strip(' ') for i in items_list) 
                newName = '; '.join(i for i in newIngList if i is not None)
                therapyName.append(newName)#5
                curie,label = getCurie(newName, name_resolver_params)
                preferred_curie, preferred_label = preferRXCUI(curie, label) #prefer RXCUI labels only if combination therapy.
                curie_ID.append(preferred_curie) #6
                curie_label.append(preferred_label) #7
                ingredient_curies.append(getIngredientCuries(newIngList, name_resolver_params)) #8
            
            else: #Single Component Therapy Handling
                combination_therapy.append("False")#4
                item = removeCationsAnionsAndBasicTerms(item, desalting_params)
                itemStatuses = getAllStatuses(rawdata,originalItem)
                therapyName.append(item) #5
                curie,label = getCurie(item, name_resolver_params) 
                curie_ID.append(curie[0])#6
                curie_label.append(label[0])#7
                ingredient_curies.append("NA")#8

    equiv_ids = get_equivalent_ids(curie_ID)

    data = pd.DataFrame({'single_ID':curie_ID, 
                        'ID_Label':curie_label, 
                        'Name_Orange_Book':name_in_orange_book,
                        'Therapy_Name':therapyName, 
                        'Combination_Therapy':combination_therapy, 
                        'Ingredient_IDs':ingredient_curies,
                        'Available_USA':available_USA,
                        'Equivalent_IDs': equiv_ids,
                        })

    data = add_approval_tags(data, approval_tags_name)
    
    return data


def generate_raw_list(rawdata: pd.DataFrame, exclusions: pd.DataFrame) -> list[str]:
    """
    Args:
        rawdata(pd.DataFrame): raw FDA data
        exclusions(pd.DataFrame): manually curated list of exclusions from FDA list.
    Returns:
        list[str]: a raw list of all of the drugs in the FDA book
    """
    drugNames = rawdata.Ingredient
    exclusions_names = exclusions['name']
    drugList = set(drugNames).difference(exclusions_names)
    return drugList

def generate_raw_ema_list(rawdata: pd.DataFrame, exclusions: pd.DataFrame) -> list[str]:
    """
    Args:
        rawdata(pd.DataFrame): raw FDA data
        exclusions(pd.DataFrame): manually curated list of exclusions from FDA list.
    Returns:
        list[str]: a raw list of all of the drugs in the FDA book
    """
    humanDrugs = rawdata[rawdata['Category']=='Human']
    approvedDrugs = humanDrugs[humanDrugs['Authorisation status']=='Authorised']
    drugnames = list(approvedDrugs['International non-proprietary name (INN) / common name'])
    exclusions_names = exclusions['name']
    drugnames = list(i.upper() if type(i)==str else i for i in drugnames)
    drugList = set(drugnames).difference(exclusions_names)
    return drugList



def generate_ob_list(rawdata: pd.DataFrame, exclusions: pd.DataFrame, split_exclusions: pd.DataFrame, desalting_params: dict, name_resolver_params: dict, approval_tag_name: str) -> pd.DataFrame:
    """
    Args:
        rawdata (pd.DatFrame): raw FDA orange book data from products file.
        exclusions: items selected by medical team for exclusion for various reasons (diagnostic/contrast/radiolabel, water, other compounds inviable for repurposing)
        split_exclusions: items containing delimiters that would normally cause the item to be split but should actually remain as a single item.
        desalting_params: parameters used for removing inactive cations, anions, and other terms from active ingredient names
        name_resolver_params: parameters used for accessing the RENCI name resolving service to acquire IDs for each compound.

    Returns:
        pd.DataFrame: a drug list containing all of the FDA-approved small-molecule therapeutic compounds, their approval statuses, and connection to their individual components when they are combination therapies.

    """
    splitExclusions = set(list(split_exclusions['name']))
    drugList = generate_raw_list(rawdata, exclusions)
    data = generate_ob_df(drugList, desalting_params, name_resolver_params, rawdata, splitExclusions, approval_tag_name)
    return data


def isCombinationTherapy_ema(item: str, exclusions: list[str]) -> bool:
    if type(item)!=float and (("," in item) or ("/" in item) or ("AND" in item)) and item.upper not in exclusions:
        return True
    return False


def split_therapy_ema(combination_therapy_name):
    """
    Args: 
        combination_therapy_name (str): full combination therapy string including delimiters.

    Returns:
        list[str]: a list with the delimiters and whitespace stripped.

    """
    ingList = re.split(', | / | AND ', combination_therapy_name)
    items_list = list(set(ingList))
    items_list.sort()
    return [x.strip() for x in items_list]



def generate_ema_df(drugList: list[str], split_exclusions: list[str], desalting_params: dict, name_resolver_params: dict, approval_tags_name:str) -> pd.DataFrame:
    Approved_EMA = []
    combination_therapy = []
    therapyName = []
    name_in_ema = []
    curie_ID = []
    curie_label = []
    ingredientCuriesList = []

    for index, item in tqdm(enumerate(list(drugList)), total=len(drugList)):
        if not testing or testing and index < limit:
            name_in_ema.append(item)#1
            #Approved_EMA.append("True")#2

            if isCombinationTherapy_ema(item, split_exclusions) and item not in split_exclusions:
                item_curie_list = []
                combination_therapy.append("True")#3
                items_list = split_therapy_ema(item)
                new_therapies = list(i for i in items_list if i not in drugList)
                for i in new_therapies:
                    drugList.add(i)
                newIngList = list(removeCationsAnionsAndBasicTerms(i.strip(), desalting_params).strip(' ') for i in items_list) 
                newName = '; '.join(i for i in newIngList if i is not None)
                therapyName.append(newName) #4

                curie,label = getCurie(newName, name_resolver_params)
                preferred_curie, preferred_label = preferRXCUI(curie, label) #prefer RXCUI labels only if combination therapy.
                curie_ID.append(preferred_curie) #5
                curie_label.append(preferred_label) #6

                item_curie_list = []
                for i in newIngList:
                    curie, label = getCurie(i, name_resolver_params)
                    item_curie_list.append(curie[0])
                
                ingredientCuriesList.append(item_curie_list)#7

            else:
                combination_therapy.append("False")#3
                therapyName.append(removeCationsAnionsAndBasicTerms(item.strip(), desalting_params))#4
                curie,label = getCurie(item, name_resolver_params)
                curie_ID.append(curie[0])#5
                curie_label.append(label[0])#6
                ingredientCuriesList.append("NA")#7

    equiv_ids = get_equivalent_ids(curie_ID)

    data = pd.DataFrame({'single_ID':curie_ID,
                     'ID_Label':curie_label,
                     'Name_EMA':name_in_ema,
                     'Therapy_Name':therapyName, 
                     'Combination_Therapy':combination_therapy, 
                     'Ingredient_IDs':ingredientCuriesList,
                     'Equivalent_IDs':equiv_ids,
                     })

    data = add_approval_tags(data, approval_tags_name)

    return data



def generate_ema_list(rawdata: pd.DataFrame, ema_exclusions: pd.DataFrame, ema_split_exclusions: pd.DataFrame, desalting_params: dict, name_resolver_params: dict, approval_tag_name: str) -> pd.DataFrame:
    splitExclusions = set(list(ema_split_exclusions['name']))
    #print(f"Split exclusions: {splitExclusions}")
    drugList = generate_raw_ema_list(rawdata, ema_exclusions)
    data = generate_ema_df(drugList, splitExclusions, desalting_params, name_resolver_params, approval_tag_name)
    return data


def generate_raw_pmda_list(rawdata: pd.DataFrame, exclusions: pd.DataFrame) -> list[str]:
    exclusions_names = exclusions['name']
    drugList = rawdata['Active Ingredient (underlined: new active ingredient)']
    for idx, item in enumerate(drugList):
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
            drugList[idx] = item
        except:
            print("encountered problem with ", item)
            drugList[idx]="error"
    return list(set(drugList).difference(exclusions_names))


def is_combination_therapy_pmda(item: str, split_exclusions: pd.DataFrame) -> bool:
    split_exc = split_exclusions['name']
    if type(item)!=float and (("," in item) or ("/" in item) or (" AND " in item) or (";" in item)) and item not in split_exc:
        return True
   
    return False


def split_therapy_pmda(item: str):
    """
    Args: 
        item (str): full combination therapy string including delimiters.

    Returns:
        list[str]: a list with the delimiters and whitespace stripped.

    """
    ingList = re.split(' , |,|/| \ | AND |; ', item)
    items_list = list(set(ingList))
    items_list.sort()
    return [x.strip() for x in items_list]




def generate_pmda_df(drugList: pd.DataFrame, split_exclusions: pd.DataFrame, desalting_params: dict, name_resolver_params: dict, approval_tags_name: str) -> pd.DataFrame:
    Approved_Japan = []
    combination_therapy = []
    therapyName = []
    name_in_pmda_list = []
    curie_ID = []
    curie_label = []
    ingredient_curies = []
    for index, item in tqdm(enumerate(drugList), total=len(drugList)):
        if not testing or testing and index < limit:
            name_in_pmda_list.append(item) #1
            Approved_Japan.append("True") #2
            
            if is_combination_therapy_pmda(item, split_exclusions) and item not in split_exclusions:
                newIngredientList = []
                combination_therapy.append("True") #3
                items_list = split_therapy_pmda(item)
                new_therapies = list(i for i in items_list if i not in drugList)
                for i in new_therapies:
                    drugList.append(i)
                newIngList = list(removeCationsAnionsAndBasicTerms(i.strip(), desalting_params).strip(' ') for i in items_list) 
                newName = '; '.join(i for i in newIngList if i is not None)
                therapyName.append(newName)#4
                curie,label = getCurie(newName, name_resolver_params)
                preferred_curie, preferred_label = preferRXCUI(curie, label) #prefer RXCUI labels only if combination therapy.
                curie_ID.append(preferred_curie) #5
                curie_label.append(preferred_label) #6
                item_curie_list = []
                for i in newIngList:
                    curie, label = getCurie(i, name_resolver_params)
                    item_curie_list.append(curie[0])
                
                ingredient_curies.append(item_curie_list)#7
            else:
                combination_therapy.append("False")
                newName = removeCationsAnionsAndBasicTerms(item.upper().strip(), desalting_params)
                therapyName.append(newName) #3
                curie, label = getCurie(newName, name_resolver_params) #4
                curie_ID.append(curie[0]) #5
                curie_label.append(label[0]) #6
                ingredient_curies.append("NA") #7
    equiv_ids = get_equivalent_ids(curie_ID)

    data = pd.DataFrame({'single_ID':curie_ID,
                        'ID_Label':curie_label,
                        'Name_PMDA':name_in_pmda_list,
                        'Therapy_Name':therapyName,
                        'Combination_Therapy':combination_therapy, 
                        'Ingredient_IDs':ingredient_curies,
                        'Equivalent_IDs':equiv_ids,
                        })

    data = add_approval_tags(data, approval_tags_name)

    return data


def generate_pmda_list(rawdata: pd.DataFrame, exclusions: pd.DataFrame, split_exclusions: pd.DataFrame, desalting_params: dict, name_resolver_params: dict, approval_tags_name: str) -> pd.DataFrame:
    drugList = generate_raw_pmda_list(rawdata, exclusions)
    data = generate_pmda_df(drugList, split_exclusions, desalting_params, name_resolver_params, approval_tags_name)
    return data


def build_drug_list(fda_list: pd.DataFrame, ema_list: pd.DataFrame, pmda_list: pd.DataFrame) -> pd.DataFrame:
    data = merge_lists(fda_list, ema_list, pmda_list)
    return data




def merge_lists(fda_list: pd.DataFrame, ema_list: pd.DataFrame, pmda_list: pd.DataFrame,):
    df1 = pd.merge(ema_list, pmda_list, on="single_ID", how="outer")
    df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]
    df1 = merge_columns('Therapy_Name_x', 'Therapy_Name_y', df1, 'Therapy_Name')
    df1 = merge_columns('ID_Label_x', 'ID_Label_y', df1, 'ID_Label')
    df1 = merge_columns('Combination_Therapy_x', 'Combination_Therapy_y', df1, 'Combination_Therapy')
    df1 = merge_columns('Ingredient_IDs_x', 'Ingredient_IDs_y', df1, 'Ingredient_IDs')
    df1 = merge_columns('Equivalent_IDs_x', 'Equivalent_IDs_y', df1, 'Equivalent_IDs')

    df1 = merge_identical_subcolumns('Combination_Therapy', df1, "|")
    df1 = merge_identical_subcolumns('Therapy_Name', df1, "|")
    df1 = merge_identical_subcolumns('ID_Label', df1, "|")


    df2 = pd.merge(df1, fda_list, on="single_ID", how="outer")
    df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
    df2 = merge_columns('ID_Label_x', 'ID_Label_y', df2, 'ID_Label')
    df2 = merge_columns('Therapy_Name_x', 'Therapy_Name_y', df2, 'Therapy_Name')
    df2 = merge_columns('Combination_Therapy_x', 'Combination_Therapy_y', df2, 'Combination_Therapy')
    df2 = merge_columns('Ingredient_IDs_x', 'Ingredient_IDs_y', df2, 'Ingredient_IDs')
    df2 = merge_columns('Equivalent_IDs_x', 'Equivalent_IDs_y', df2, 'Equivalent_IDs')

    for idx, row in df2.iterrows():
        if not row['approved_eu']== True:
            df2.loc['approved_eu',idx] = False
        if not row['approved_japan',idx]== True:
            df2.loc['approved_japan',idx] = False
        if not row['approved_usa'] == True:
            df2.loc['approved_usa',idx] = False

    df2 = merge_identical_subcolumns('Combination_Therapy', df2, "|")
    df2 = merge_identical_subcolumns('Therapy_Name', df2, "|")
    df2 = merge_identical_subcolumns('ID_Label', df2, "|")

    return df2
    #df2.to_csv("drugList.tsv", sep='\t')    



def merge_columns (name1, name2, df, newname):
    df[newname]=df.apply(lambda x:'%s|%s' % (x[name1],x[name2]),axis=1)
    df.drop(name1,axis=1,inplace=True)
    df.drop(name2, axis=1,inplace=True)

    for idx, row in df.iterrows():
        if 'nan' in row[newname]:
            df.loc[newname,idx] = row[newname].replace('nan', "").replace('|','')
    return df

def merge_identical_subcolumns(colname, df, delimiter):
    for idx, row in df.iterrows():
        subColumns = row[colname].split(delimiter)
        for idx2, i in enumerate(subColumns):
            subColumns[idx2] = i.strip()
        if len(subColumns)==2:
            if subColumns[0] == subColumns[1]:
                df.loc[colname,idx] = subColumns[0]

    return df

def generate_tag(drug_list:List, model_params:Dict)-> List:
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

def enrich_drug_list(drug_list:List, params:Dict)-> pd.DataFrame:
    """Node enriching existing drug list with llm-generated tags.
    
    Args:
        drug_list: pd.DataFrame - merged drug_list with drug names column that will be used for tag generation
        params: Dict - parameters dictionary specifying tag names, column names, and model params
    Returns
        pd.DataFrame with x new tag columns (where x corresponds to number of tags specified in params)
    """
    for tag in params.keys():
        print(f"applying tag: \'{tag}\' to drug list")
        input_col = params[tag].get('input_col')
        output_col = params[tag].get('output_col')
        model_params = params[tag].get('model_params')
        drug_list[output_col] = generate_tag(drug_list[input_col], model_params)
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


































































































































#########################################
##### GENERAL LIST BUILDER HERE #########
#########################################
def remove_manually_excluded_drugs(list_in: pd.DataFrame, exclusions: list[str], drug_name_column: str) -> pd.DataFrame:
    indices_to_drop = []
    for idx, row in list_in.iterrows():
        if row[drug_name_column] in exclusions:
            indices_to_drop.append(idx)
    return list_in.drop(indices_to_drop)


def preprocess_ema(rawdata: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        rawdata(pd.DataFrame): raw FDA data
        exclusions(pd.DataFrame): manually curated list of exclusions from FDA list.
    Returns:
        list[str]: a raw list of all of the drugs in the FDA book
    """
    indices_to_drop = []
    for idx, row in rawdata.iterrows():
        if (row['Category'] != "Human") or (row['Authorisation status'] != "Authorised"):
            indices_to_drop.append(idx)
    new_df = rawdata.drop(indices_to_drop)
    return new_df

def remove_illegal_characters_openpyxl(input_list: pd.DataFrame):
    return input_list.applymap(lambda x: x.encode('unicode_escape').
                 decode('utf-8') if isinstance(x, str) else x)


def create_standardized_columns(df_in: pd.DataFrame, drug_name_column:str, approval_date_column:str) -> pd.DataFrame:
    df_in.rename(
        columns={drug_name_column:'drug_name',
                 approval_date_column:'approval_date',
                 },
        inplace=True
        )
    # remove illegal unicode characters from list:
    df_in = remove_illegal_characters_openpyxl(df_in)
    return df_in

def is_combination_therapy(item: str, delimiters: list[str], exclusions: list[str]) -> bool:
    if type(item)==float:
        return False
    if item in exclusions:
        return False
    for i in delimiters:
        if i in item:
            return True
    return False

def tag_combination_therapies(inputList: pd.DataFrame, delimiters: list[str], exclusions: list[str])->pd.DataFrame:
    combination_therapy = []
    for idx, row in tqdm(inputList.iterrows(), total=len(inputList)):
        combination_therapy.append(is_combination_therapy(row['drug_name'], delimiters, exclusions))
    inputList['combination_therapy'] = combination_therapy
    return inputList


def identify(name, params):
    """
    Args:
        name (str): string to be identified
        params (tuple): name resolver parameters to feed into get request
    
    Returns:
        resolvedName (list[str]): IDs most closely matching string.
        resolvedLabel (list[str]): List of labels associated with respective resolvedName.

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
            return "Error", "Error"
    return resolvedName[0], resolvedLabel[0]

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
            item_curie, item_label = identify(drug_name, params)
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
    original_dataframe[column_name]=True
    return original_dataframe

def multi_split(inString, delims):
    result = [inString]
    for delim in delims:
        result = [substr for s in result for substr in s.split(delim)]
    return [s for s in result if s]

def add_ingredients(input_list: pd.DataFrame, delimiters: list[str]):
    ingredients_list = []
    for idx,row in tqdm(input_list.iterrows(), total=len(input_list)):
        if row['combination_therapy']==True:
            ingredients_list.append(multi_split(row['drug_name'], delimiters))
        else:
            ingredients_list.append(None)

    input_list["ingredients_list"]=ingredients_list
    return input_list

def add_row (original_list: pd.DataFrame, columns: dict) -> pd.DataFrame:
    original_list = pd.concat([original_list, pd.DataFrame(columns, index = [0])], ignore_index=True)
    return original_list

def add_unlisted_single_ingredients(input_list: pd.DataFrame) -> pd.DataFrame:
    ingList = list(input_list['drug_name'])
    for idx, row in tqdm(input_list.iterrows(), total=len(input_list), desc="adding unlisted single therapies"):
        if row['combination_therapy']==True:
            for item in string_to_list(row['ingredients_list']):
                if item not in ingList:
                    new_columns = {
                        'drug_name':item, 
                        'approval_date':row['approval_date'],
                        'combination_therapy':False
                        }
                    input_list = add_row(input_list, new_columns)
                    ingList = list(input_list['drug_name'])
    return input_list

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
                    curie,label = identify(ingredient, nameres_params)
                    curr_row_ingredient_ids.append(curie)
                    cache[ingredient]=curie
            ingredient_ids_list.append(curr_row_ingredient_ids)   
    input_list["ingredient_ids"] = ingredient_ids_list
    return input_list

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

def add_SMILES_strings(in_list: pd.DataFrame) -> pd.DataFrame:
    smiles_strings = []
    for idx, row in tqdm(in_list.iterrows(), total = len(in_list), desc="adding SMILES strings through PUBCHEM"):
        
        if "|" in row['Equivalent_IDs']:
            smiles_strings.append(None)
        else:
            not_found = True
            for id in string_to_list(row['Equivalent_IDs']):
                if "PUBCHEM" in id and not_found:
                    #print(id)
                    not_found = False
                    pc_index = id.replace("PUBCHEM.COMPOUND:","")
                    #print(f"found pubchem compound {pc_index}")
                    smiles_strings.append(get_smiles_from_pubchem(int(pc_index)))
                    continue
            if not_found:
                smiles_strings.append(None)

    in_list['SMILES_string']=smiles_strings
    return in_list


#def build_list(input_data: pd.DataFrame, delimiters: list[str], exclusions: list[str], split_exclusions: list[str], id_params: dict) -> pd.DataFrame:
    # data_import_params = {}
    # drug_list_base = import_list(input_data, data_import_params)
    # drug_list_with_combination_therapy_tags = tag_combination_therapies(drug_list_base, delimiters, exclusions)
    # drug_list_with_ids = identify_drugs(drug_list_with_combination_therapy_tags, id_params)
    # drug_list_with_ingredients = add_ingredients(drug_list_with_ids, delimiters)
    # drug_list_with_ingredient_ids = add_ingredient_ids(drug_list_with_ingredients, ";")
    
    # drug_list_with_approval_tags = add_approval_tags(drug_list_with_ingredients, params['approval_tag_name'])
    # drug_list_final = pd.DataFrame()
    # return drug_list_final

