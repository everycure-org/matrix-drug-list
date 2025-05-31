
import pandas as pd
from tqdm import tqdm

def compare_medi_robokop(medi_list: pd.DataFrame, robokop_list:pd.DataFrame) -> pd.DataFrame:
    rk_edge_set = []
    
    for idx, row in tqdm(robokop_list.iterrows(), total=len(robokop_list), desc="extracting robokop unique d-d edges"):
        rk_edge_set.append("|".join([row['Treatment'], row['Disease']]))

    print(f"{len(rk_edge_set)} edges found in robokop")
    print(f"{len(set(rk_edge_set))} unique")

    for idx, row in tqdm(medi_list)
    return None