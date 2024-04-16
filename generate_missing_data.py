import itertools
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd

def generate_missing_data(args, X_df, y_df):

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X_df_scaled = pd.DataFrame(X_scaled, columns=X_df.columns)
    y_df = y_df.reset_index(drop=True)

    complete_df = pd.concat([X_df_scaled, y_df], axis=1)

    new_X_rows = []
    new_Z_rows = []

    if args.missing_pattern == 'single':
        for index, row in complete_df.iterrows():
            new_X_row = row.copy()
            new_X_row.iloc[args.col_to_remove] = 0
            new_X_rows.append(new_X_row)
            new_Z_rows.append(complete_df.loc[index])

    elif args.missing_pattern == 'multiple':
        features = [col for col in complete_df.columns if col != "target"]
        for index, row in complete_df.iterrows():
            for r in range(1, args.max_remove_count + 1):  
                for subset in itertools.combinations(features, r):
                    new_X_row = row.copy()
                    new_X_row[list(subset)] = 0
                    new_X_rows.append(new_X_row)
                    new_Z_rows.append(complete_df.loc[index])

    elif args.missing_pattern == 'random':
        features = [col for col in complete_df.columns if col != "target"]
        feature_combinations = []

        for r in range(1, args.max_remove_count + 1):
            feature_combinations += list(itertools.combinations(features, r))
            
        for index, row in complete_df.iterrows():
            random_combinations = random.sample(feature_combinations, args.new_num_per_origin)
            for subset in random_combinations:
                new_X_row = row.copy()
                new_X_row[list(subset)] = 0
                new_X_rows.append(new_X_row)
                new_Z_rows.append(complete_df.loc[index])

    X_df = pd.concat(new_X_rows, ignore_index=True, axis=1).T
    Z_df = pd.concat(new_Z_rows, ignore_index=True, axis=1).T
        
    if args.include_complete:
        X_df = pd.concat([X_df, complete_df], ignore_index=True)
        Z_df = pd.concat([Z_df, complete_df], ignore_index=True)

    if args.dataset_name != 'boston':   
        y_df = X_df["target"].astype(int)
    X_df = X_df.drop("target", axis=1)
    Z_df = Z_df.drop("target", axis=1)
    
    return X_df, Z_df, y_df