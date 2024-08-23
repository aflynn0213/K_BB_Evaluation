import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


if __name__ == "__main__":
    df = pd.read_csv("r2pitch.csv")
    df.columns = df.columns.str.lower()
    df.set_index(['season','name'],inplace=True)
    feat_cols = ['swstr%'] + list(df.loc[:,'stuff+':'cstr%'].columns)
    print(df[df[feat_cols].isna().any(axis=1)])
    print(feat_cols)
    df = pd.concat([df['k%'],df[feat_cols]],axis=1)
    players_2021 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2021].unique()
    players_2022 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2022].unique()
    players_2023 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2023].unique()
    players_2024 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2024].unique()

    common_players = set(players_2021).intersection(players_2022)
    common_players_2 = set(players_2022).intersection(players_2023)
    common_players_3 = set(players_2023).intersection(players_2024)
    
    # Step 2: Filter the data for common players from 2021 and 2022
    x_1 = df.loc[(2021, list(common_players)), "k%"]
    y_1 = df.loc[(2022, list(common_players)), "k%"]
    x_2 = df.loc[(2022, list(common_players_2)), "k%"]
    y_2 = df.loc[(2023, list(common_players_2)), "k%"]
    x_3 = df.loc[(2023, list(common_players_3)), "k%"]
    y_3 = df.loc[(2024, list(common_players_3)), "k%"]

    x_1 = x_1.sort_index(level='name')
    y_1 = y_1.sort_index(level='name')
    x_2 = x_2.sort_index(level='name')
    y_2 = y_2.sort_index(level='name')
    x_3 = x_3.sort_index(level='name')
    y_3 = y_3.sort_index(level='name')

    x = pd.concat([x_1,x_2,x_3])
    y = pd.concat([y_1,y_2,y_3])

    print(x.shape)
    print(y)

    print(r2_score(y,x))
'''
    # Step 3: Predict k% for the 2022 season using the 2021 data and calculate R-squared
    predictions_2022 = model.predict(data_2021_common[x_columns])
    r2_2022 = r2_score(data_2022_common['k%'], predictions_2022)
    # Step 2: Filter the data for common players from 2021 and 2022
    x_1 = df.loc[(2021, list(common_players)), :]
    x_2 = df.loc[(2022, list(common_players_2)), :]
    k_1 = df.loc[(2022, list(common_players)), "k%"]
    k_2 = df.loc[(2023, list(common_players_2)), "k%"]

    y = pd.concat([k_1,k_2])
    x = pd.concat([x_1,x_2])
    x = x[feat_cols]
    print(x)
    print(y)   '''  
