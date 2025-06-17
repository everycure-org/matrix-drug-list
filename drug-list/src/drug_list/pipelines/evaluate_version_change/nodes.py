"""
This is a boilerplate pipeline 'evaluate_version_change'
generated using Kedro 0.19.10
"""
import pandas as pd 

def store_previous_version(in_list: pd.DataFrame) -> pd.DataFrame:
    return in_list

def compare_versions(previous_list: pd.DataFrame, current_list: pd.DataFrame) -> pd.DataFrame:
    drugs_old = set(previous_list['improved_id'])
    drugs_new = set(current_list['improved_id'])

    drugs_removed = drugs_old.difference(drugs_new)
    drugs_added = drugs_new.difference(drugs_old)
    drugs_same = drugs_new.intersection(drugs_old)

    print(f"{len(drugs_removed)} drugs removed from list : {drugs_removed}")
    print(f"{len(drugs_added)} drugs added to list: {drugs_added}")
    print(f"{len(drugs_same)} drugs remain the same between versions.")

    print(previous_list)
    print(current_list)

    return pd.DataFrame({
        "drugs_added": pd.Series(list(drugs_added)),
        "drugs_removed": pd.Series(list(drugs_removed)),
        "drugs_same": pd.Series(list(drugs_same))
    })