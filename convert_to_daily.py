"""
Convert minute-level returns CSV files to daily returns.
This reduces file size significantly for GitHub storage.
"""

import pandas as pd
import glob
import os

def convert_minute_to_daily(input_file, output_file):
    """
    Convert minute-level returns to daily returns by compounding.
    
    Args:
        input_file: Path to input CSV file with minute-level returns
        output_file: Path to output CSV file with daily returns
    """
    print(f"Processing: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    # Fill NaN with 0 for compounding
    df = df.fillna(0)
    
    # Resample to daily and compound returns
    # For returns: daily_return = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    # We use resample with a custom function to compound returns
    def compound_returns(series):
        """Compound returns: (1+r1)*(1+r2)*...*(1+rn) - 1"""
        return (1 + series).prod() - 1
    
    # Resample to daily (business days) and compound returns
    df_daily = df.resample('D').apply(compound_returns)
    
    # Remove any days with all NaN (weekends/holidays if using business days)
    # Actually, let's keep all days and just fill remaining NaN with 0
    df_daily = df_daily.fillna(0)
    
    # Save to new file
    df_daily.to_csv(output_file)
    
    # Print stats
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    reduction = (1 - new_size / original_size) * 100
    
    print(f"  Original: {df.shape[0]:,} rows, {original_size:.2f} MB")
    print(f"  Daily:    {df_daily.shape[0]:,} rows, {new_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Saved to: {output_file}\n")
    
    return df_daily

def main():
    """Convert all CSV files in data directory to daily returns."""
    
    # Find all CSV files in data directory (excluding _daily.csv files)
    data_dir = "data"
    pattern = os.path.join(data_dir, "*.csv")
    csv_files = [f for f in glob.glob(pattern) if "_daily.csv" not in f]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}/")
        return
    
    print(f"Found {len(csv_files)} file(s) to convert:\n")
    
    for input_file in csv_files:
        # Create output filename by inserting _daily before .csv
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_daily.csv"
        
        try:
            convert_minute_to_daily(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}\n")
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()

