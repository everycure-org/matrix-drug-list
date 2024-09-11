"""
This is a boilerplate pipeline 'generate_orange_book_list'
generated using Kedro 0.19.8
"""
import pandas as pd
pd.options.mode.chained_assignment = None  #default='warn'
import numpy as np
import difflib as dl
import psycopg2 as pg
import re
import requests
from io import StringIO

def generate_ob_list(rawdata: pd.DataFrame, exclusions: pd.DataFrame, split_exclusions: pd.DataFrame) -> pd.DataFrame:


    def preferRXCUI(curieList, labelList):
        for idx, item in enumerate(curieList):
            if "RXCUI" in item:
                return item, labelList[idx]
        return curieList[0], labelList[0]           

    def getCurie(name):
        itemRequest = 'https://name-resolution-sri.renci.org/lookup?string=' + name + '&autocomplete=false&offset=0&limit=10&biolink_type=ChemicalOrDrugOrTreatment'
        returned = (pd.read_json(StringIO(requests.get(itemRequest).text)))
        resolvedName = returned.curie
        resolvedLabel = returned.label
        return resolvedName, resolvedLabel

    def getCombinationTherapiesAndSingleTherapiesLists(orangebook, exclusions):
        obCombinationTherapies = []
        obSingleTherapies = []
        ingredientList = set(list(orangebook.Ingredient))
        for item in ingredientList:
            if (";" in item) or (" AND " in item) or ("W/" in item):
                obCombinationTherapies.append(item)
            else:
                obSingleTherapies.append(item.strip())
        return list(set(obCombinationTherapies)), list(set(obSingleTherapies))

    def getAllStatuses(orangebook, item):
        indices = [i for i, x in enumerate(orangebook['Ingredient']) if x == item]
        return list(orangebook['Type'][indices])

    def getMostPermissiveStatus(statusList):
        if "OTC" in statusList:
            return "OTC"
        elif "RX" in statusList:
            return "RX"
        elif "DISCN" in statusList:
            return "DISCONTINUED"
        else:
            return "UNSURE"

    def isBasicCation(item):
        basic_cations = ['FERROUS', 
                        'CALCIUM', 
                        'SODIUM', 
                        'MAGNESIUM', 
                        'MANGANESE', 
                        'POTASSIUM', 
                        'ALUMINUM', 
                        'TITANIUM', 
                        'COPPER', 
                        'CUPRIC', 
                        'LYSINE']
        
        if item in basic_cations:
            return True

        return False

    def isBasicAnion(item):
        basic_anions = ['CHLORIDE', 
                        'DIOXIDE', 
                        'OXIDE', 
                        'ACETATE', 
                        'SULFATE', 
                        'PHOSPHATE', 
                        'HYDROXIDE', 
                        'HYDROCHLORIDE',
                        'CITRATE', 
                        'DIACETATE', 
                        'TRIACETATE', 
                        'ADIPATE', 
                        'TARTRATE', 
                        'BITARTRATE', 
                        'FUMARATE', 
                        'HEMIFUMARATE',
                        'MALEATE', 
                        'BROMIDE', 
                        'MEGLUMINE', 
                        'BICARBONATE', 
                        'MESYLATE', 
                        'DISULFIDE', 
                        'FLUORIDE', 
                        'GLYCEROPHOSPHATE']

        if item in basic_anions:
            return True

        return False

    def isOtherBasicTerm(item):
        other_identifiers = ['HYDRATE', 
                            'DIHYDRATE', 
                            'MONOHYDRATE', 
                            'TRIHYDRATE', 
                            'ANHYDROUS', 
                            'MONOBASIC', 
                            'DIBASIC', 
                            'LYSINE', 
                            'ARGININE',
                            'HEPTAHYDRATE']

        if item in other_identifiers:
            return True
            
        return False

    def isBasicSaltOrMetalOxide(inString):
        components = inString.strip().split()
        
        for item in components:
            item = item.replace(';', '').replace(',','')
            if not isBasicCation(item) and not isBasicAnion(item) and not isOtherBasicTerm(item):
                return False
                
        return True

    def removeCationsAnionsAndBasicTerms(ingredientString):
        if not isBasicSaltOrMetalOxide(ingredientString):
            components = ingredientString.strip().split()
            for ind,i in enumerate(components):
                if isBasicAnion(i) or isBasicCation(i) or isOtherBasicTerm(i):
                    components[ind] = ''
            newString = ''
            for i in components:
                newString = newString + i + " "
            newString = newString[:-1]
            return newString
        return ingredientString
            
    splitExclusions = set(list(split_exclusions['name']))
    obCombinationTherapies, obSingleTherapies = getCombinationTherapiesAndSingleTherapiesLists(rawdata, splitExclusions)
    print(len(set(obCombinationTherapies)), " combination therapeutics.")
    print(len(set(obSingleTherapies)), " single-ingredient therapeutics.")
    obSingleSet = set(obSingleTherapies)
    print("splitting combination therapies (currently ", len(obSingleSet), " items in list)")
    
    exclusions_names = exclusions['name']
    Approved_USA = []
    combination_therapy = []
    therapyName = []
    name_in_orange_book = []
    available_USA = []
    curie_ID = []
    curie_label = []
    ingredient_curies = []

    drugList = list(set(obCombinationTherapies + obSingleTherapies).difference(exclusions_names))

    labelDict = {}
    idDict = {}
    for index, item in enumerate(drugList):
        originalItem = item
        
        if originalItem in obCombinationTherapies:
            name_in_orange_book.append(item)#1
            Approved_USA.append("True")#2
            combination_therapy.append("True")#3
            available_USA.append(getMostPermissiveStatus(getAllStatuses(orangebook,item)))#4
            print("item",index, ":", item)
        
            newIngList = []
            ingList = re.split('; | ; | AND | W/ ', item)
            ingredientCuriesList = []


            itemsList = list(set(ingList))
            itemsList.sort()
            
            for i in itemsList:
                curie, label = getCurie(i)
                ingredientCuriesList.append(curie[0])
                if i not in obSingleTherapies:
                    drugList.append(i.strip())
                    obSingleTherapies.append(i.strip())
                #print("old name: ", i, "; new name: ", removeCationsAnionsAndBasicTerms(i))
                newIngList.append(removeCationsAnionsAndBasicTerms(i)) #5

            ingredient_curies.append(ingredientCuriesList) #6
            newName = ""

            
            for i in newIngList:
                if i is not None:
                    newName += i + "; "
            newName = newName[:-2]
            therapyName.append(newName)#9
            
            curie,label = getCurie(newName)
            preferred_curie, preferred_label = preferRXCUI(curie, label) #prefer RXCUI labels only if combination therapy.
            curie_ID.append(preferred_curie) #7
            curie_label.append(preferred_label) #8
            
            
        elif originalItem in obSingleTherapies:
            item = removeCationsAnionsAndBasicTerms(item)
            itemStatuses = getAllStatuses(orangebook,originalItem)
            name_in_orange_book.append(originalItem)
            therapyName.append(item)
            Approved_USA.append("True")
            combination_therapy.append("False")
            available_USA.append(getMostPermissiveStatus(getAllStatuses(orangebook,originalItem)))
            print("item ", index, ": ", item)
            curie,label = getCurie(item)
            curie_ID.append(curie[0])
            curie_label.append(label[0])
            ingredient_curies.append("NA")

    print(len(obSingleTherapies), "single-component therapies after splitting")
    print(len(obSingleTherapies + obCombinationTherapies), " total therapies after splitting")
    print(len(therapyName), " therapies after exclusions")

    data = pd.DataFrame({'single_ID':curie_ID, 
                        'ID_Label':curie_label, 
                        'Name_Orange_Book':name_in_orange_book,
                        'Therapy_Name':therapyName, 
                        'Approved_USA': Approved_USA, 
                        'Combination_Therapy':combination_therapy, 
                        'Ingredient_IDs':ingredient_curies,
                        'Available_USA':available_USA,})

    return data