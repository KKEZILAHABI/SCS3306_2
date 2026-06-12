import pandas as pd
import numpy as np

def get_exploration_report(df):
    """
    Analyzes a DataFrame and returns a summary of NaNs, Zeros, and Blanks per column.
    """
    results = []
    
    for col in df.columns:
        # 1. Count NaNs (Null values)
        nan_count = df[col].isna().sum()
        
        # 2. Count Zeros (Checks for both numeric 0 and string '0')
        zero_count = (df[col] == 0).sum() + (df[col] == '0').sum()
        
        # 3. Count Blanks (Empty strings or strings with only spaces)
        # We only apply this to object/string columns to prevent TypeErrors
        if df[col].dtype in ['object', 'string']:
            blank_count = df[col].astype(str).str.strip().eq('').sum()
        else:
            blank_count = 0
            
        # Calculate total potential missing/invalid data
        total_issues = nan_count + zero_count + blank_count
        
        results.append({
            'Column': col,
            'Data Type': df[col].dtype,
            'NaNs': nan_count,
            'Zeros': zero_count,
            'Blanks': blank_count,
            'Total Issues': total_issues
        })
        
    # Convert results list to a DataFrame
    summary_df = pd.DataFrame(results)
    
    # Optional: Filter out columns that are perfectly clean
    # summary_df = summary_df[summary_df['Total Issues'] > 0]
    
    # Sort by the most problematic columns first
    summary_df = summary_df.sort_values(by='Total Issues', ascending=False).reset_index(drop=True)
    
    return summary_df

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # 1. Load your actual data
    df = pd.read_csv('carclaims.csv')
    
    # 2. Run the function
    report = get_exploration_report(df)
    
    # 3. View the results
    print("Data Exploration Report:")
    print("-" * 60)
    print(report)