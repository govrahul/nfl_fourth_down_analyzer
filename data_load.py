import pandas as pd
import glob
import re
import os


# usage: df = load_fourth_down_data('Data/play_by_play_*.csv')
def load_fourth_down_data(path_pattern='play_by_play_*.csv', download=True):
    dfs = []

    for file in glob.glob(path_pattern):
        match = re.search(r'(\d{4})', file)
        if not match:
            raise ValueError(f"Could not extract year from filename: {file}")
        year = int(match.group(1))

        df = pd.read_csv(file)

        required_cols = {'down', 'play_type'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{file} missing required columns")

        df = df[
            (df['down'] == 4.0) &
            (~df['play_type'].isin(['no_play', 'qb_kneel']))
        ]

        df['season'] = year
        
        df['go'] = df['play_type'].isin(['run', 'pass']).astype(int)

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    if download:
        output_dir = os.path.dirname(path_pattern)

        if output_dir == "":
            output_path = "fourth_down_data.csv"
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "fourth_down_data.csv")

        combined_df.to_csv(output_path, index=False)

    return combined_df
