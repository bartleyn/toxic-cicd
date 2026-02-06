import pandas as pd
import subprocess



def load_dataset_csv(path: str, text_col: str, label_col: str, num_rows: int = None) -> pd.DataFrame:
    
    df = pd.read_csv(path, nrows=num_rows)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {text_col} and/or {label_col} not found in the dataset.")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)

    return df[[text_col, label_col]]

def get_git_sha() -> str:
    '''
    Returns the current git commit SHA
    '''
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        sha = 'unknown'
    return sha