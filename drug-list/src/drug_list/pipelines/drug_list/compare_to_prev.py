import pandas as pd 
from tqdm import tqdm
from . import nodes

def compare(previous_list: pd.DataFrame, current_list: pd.DataFrame) -> pd.DataFrame:
    
    drugs_old = set(previous_list['improved_id'])
    drugs_new = set(current_list['improved_id'])

    drugs_removed = drugs_old.difference(drugs_new)
    drugs_added = drugs_new.difference(drugs_old)
    drugs_same = drugs_new.intersection(drugs_old)

    print(f"{len(drugs_removed)} drugs removed from list : {drugs_removed}")
    print(f"{len(drugs_added)} drugs added to list: {drugs_added}")
    print(f"{len(drugs_same)} drugs remain the same between versions.")

    #print(previous_list)
    #print(current_list)

    drugs_added_labels = (nodes.normalize(item)[1] for item in tqdm(drugs_added))
    drugs_removed_labels = (nodes.normalize(item)[1] for item in tqdm(drugs_removed))
    drugs_same_labels = (nodes.normalize(item)[1] for item in tqdm(drugs_same))

    return pd.DataFrame({
        "drugs_added": pd.Series(list(drugs_added)),
        "added_label": pd.Series(list(drugs_added_labels)),
        "drugs_removed": pd.Series(list(drugs_removed)),
        "removed_label": pd.Series(list(drugs_removed_labels)),
        "drugs_same": pd.Series(list(drugs_same)),
        "same_label":pd.Series(list(drugs_same_labels))
    })

def store_previous_version(in_list: pd.DataFrame) -> pd.DataFrame:
    return in_list