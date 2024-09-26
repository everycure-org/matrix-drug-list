import pandas as pd
pd.options.mode.chained_assignment = None  #default='warn'
import numpy as np
import difflib as dl
import psycopg2 as pg
import re
import requests
from io import StringIO
from typing import List



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


def getCurie(name, params):
    """
    Args:
        name (str): string to be identified
        params (tuple): name resolver parameters to feed into get request
    
    Returns:
        resolvedName (list[str]): IDs most closely matching string.
        resolvedLabel (list[str]): List of labels associated with respective resolvedName.

    """

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
    while not success:
        try:
            print("Resolving ", name)
            returned = (pd.read_json(StringIO(requests.get(itemRequest).text)))
            resolvedName = returned.curie
            resolvedLabel = returned.label
            success = True
        except:
            print('name resolver error')
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


def add_row(original_dataframe: pd.DataFrame, column_values: dict) -> pd.DataFrame:
    """
    args:
        original_dataframe (pd.DataFrame): current data frame.
        column_values (dict): all of the values of the column to be added

    returns:
        pd.DataFrame: the original dataframe with a new row appended.
    """

    return None


def generate_ob_df(drugList: list[str], desalting_params, name_resolver_params, rawdata, split_exclusions ) -> pd.DataFrame:
    Approved_USA, combination_therapy, therapyName, name_in_orange_book, available_USA, curie_ID, curie_label, ingredient_curies = ([] for i in range(8))
    
    for index, item in enumerate(list(drugList)):
        print("item",index, ":", item)
        originalItem = item
        # Things that get updated in the same way whether the therapy is a combination therapy or not 
        name_in_orange_book.append(originalItem) #1
        Approved_USA.append("True") #2
        available_USA.append(getMostPermissiveStatus(getAllStatuses(rawdata,item)))#3

        if isCombinationTherapy(item, split_exclusions): # Combination Therapy Handling
            combination_therapy.append("True")#4
            items_list = split_therapy_fda(originalItem)
            new_therapies = list(i for i in items_list if i not in drugList)
            print("found new therapies: ", new_therapies)
            for i in new_therapies:
                drugList.add(i)
            newIngList = list(removeCationsAnionsAndBasicTerms(i.strip(), desalting_params).strip(' ') for i in items_list) 
            newName = '; '.join(i for i in newIngList if i is not None)
            print("new name after desalting: ", newName)
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

    data = pd.DataFrame({'single_ID':curie_ID, 
                        'ID_Label':curie_label, 
                        'Name_Orange_Book':name_in_orange_book,
                        'Therapy_Name':therapyName, 
                        'Approved_USA': Approved_USA, 
                        'Combination_Therapy':combination_therapy, 
                        'Ingredient_IDs':ingredient_curies,
                        'Available_USA':available_USA,})
    
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
    drugList = set(drugnames).difference(exclusions_names)
    return drugList


def generate_ob_list(rawdata: pd.DataFrame, exclusions: pd.DataFrame, split_exclusions: pd.DataFrame, desalting_params: dict, name_resolver_params: dict) -> pd.DataFrame:
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
    data = generate_ob_df(drugList, desalting_params, name_resolver_params, rawdata, splitExclusions)
    return data



def isCombinationTherapy_ema(item: str, exclusions: list[str]) -> bool:
    if type(item)!=float and (("," in item) or ("/" in item) or ("AND" in item)) and item not in exclusions:
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



def generate_ema_df(drugList: list[str], split_exclusions: list[str], desalting_params: dict, name_resolver_params: dict) -> pd.DataFrame:
    Approved_EMA = []
    combination_therapy = []
    therapyName = []
    name_in_ema = []
    curie_ID = []
    curie_label = []
    ingredientCuriesList = []

    for index, item in enumerate(list(drugList)):
        print(index, item)
        name_in_ema.append(item)#1
        Approved_EMA.append("True")#2

        if isCombinationTherapy_ema(item, split_exclusions):
            item_curie_list = []
            combination_therapy.append("True")#3

            items_list = split_therapy_ema(item)
            new_therapies = list(i for i in items_list if i not in drugList)
            print("found new therapies: ", new_therapies)
            for i in new_therapies:
                drugList.add(i)
            newIngList = list(removeCationsAnionsAndBasicTerms(i.strip(), desalting_params).strip(' ') for i in items_list) 
            newName = '; '.join(i for i in newIngList if i is not None)
            print("new name:", newName)
            therapyName.append(newName) #4

            curie,label = getCurie(newName, name_resolver_params)
            preferred_curie, preferred_label = preferRXCUI(curie, label) #prefer RXCUI labels only if combination therapy.

            #curie, label = preferRXCUI(getCurie(newName, name_resolver_params))
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

    data = pd.DataFrame({'single_ID':curie_ID,
                     'ID_Label':curie_label,
                     'Name_EMA':name_in_ema,
                     'Therapy_Name':therapyName, 
                     'Approved_Europe': Approved_EMA, 
                     'Combination_Therapy':combination_therapy, 
                     'Ingredient_IDs':ingredientCuriesList})
    return data


def generate_ema_list(rawdata: pd.DataFrame, ema_exclusions: pd.DataFrame, ema_split_exclusions: pd.DataFrame, desalting_params: dict, name_resolver_params: dict) -> pd.DataFrame:
    splitExclusions = set(list(ema_split_exclusions['name']))
    drugList = generate_raw_ema_list(rawdata, ema_exclusions)
    data = generate_ema_df(drugList, splitExclusions, desalting_params, name_resolver_params)

    return data



def generate_pmda_list(rawdata: pd.DataFrame, exclusions: pd.DataFrame, split_exclusions: pd.DataFrame, desalting_params: dict, name_resolver_params: dict) -> pd.DataFrame:
    return None


def build_drug_list(fda_rawdata: pd.DataFrame, ) -> pd.DataFrame:
    generate_ob_list(rawdata, exclusions, fda_ob_split_exclusions, desalting_params, name_resolver_params)
    generate_ema_list(ema_raw_data_set, ema_exclusions, ema_split_exclusions, desalting_params, name_resolver_params)