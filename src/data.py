from __future__ import annotations
import pandas as pd
from pathlib import Path

DEFAULT_FILE = Path("data/Customer_Data.csv")  # place your CSV here

NUMERIC_COLS = [
    'BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY',
    'PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX',
    'PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE'
]

def load_dataset(path: str | Path = DEFAULT_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning: drop obvious identifiers if present
    for c in ['CUST_ID']:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df