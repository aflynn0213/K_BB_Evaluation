from posixpath import split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def evaluate_model(model, X_test, y_test, target_variable):
    if 'xgb' in target_variable.lower():
        print("XGB MODEL")
    elif 'poly' in target_variable.lower():
        poly = model.named_steps['poly']
        model = model.named_steps['regressor']
        X_test = poly.transform(x_test)
    else:
        print("Model type not recognized try again")
    
    # Make predictions
    y_preds = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)
    r2 = r2_score(y_test, y_preds)

    # Print evaluation metrics
    print(f"Evaluation metrics for {target_variable}:")
    print(f"Model Type {model}:")
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2) Score:", r2)
    print()
    
    return y_preds
    
    
def polynomial_learning(X, y, target_variable):
    # Pipeline definition
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(include_bias=True)),
        ('regressor', Lasso())  # Initial regressor, will be replaced later
    ])
    
    # Parameter grid
    param_grid = {
        'regressor': [Lasso(), Ridge()],  # Regressors to try
        'poly__degree': [1, 2, 3, 4],  # Degrees to try
        'poly__interaction_only': [False],
        'regressor__alpha': [0.001, 0.01, 0.1, 0.25, 1.0, 2.5, 5, 10.0, 20]  # Alphas to try
    }

    # GridSearchCV object
    model = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=4)
    
    # Fit the model
    model.fit(X, y)
    
    # Extract best estimator and regressor
    best_estimator = model.best_estimator_
    best_regressor = best_estimator.named_steps['regressor']

    # Determine the best regressor type
    if isinstance(best_regressor, Lasso):
        print(f"Best regressor for {target_variable}: Lasso")
    elif isinstance(best_regressor, Ridge):
        print(f"Best regressor for {target_variable}: Ridge")
    else:
        print(f"Unknown best regressor for {target_variable}")

    # Extract polynomial features and regressor
    poly = best_estimator.named_steps['poly']
    regress = best_estimator.named_steps['regressor']

    # Print best score
    print(f"BEST SCORE for {target_variable}: {-model.best_score_}")

    # Transform features using the best estimator
    X_poly = poly.transform(X)
    
    # Fit the model on polynomial features
    regress.fit(X_poly, y)
    
    # Print coefficients and best parameters
    print("Coefficients:", regress.coef_)
    print(f"Best training parameters for {target_variable}: ", regress.get_params())

    return best_estimator

def split_data(df):
    df.columns = df.columns.str.lower()
    df.set_index(['season','name'],inplace=True)
    feat_cols = ['swstr%'] + list(df.loc[:,'stuff+':'cstr%'].columns)

    df[feat_cols] = df[feat_cols].replace('', np.nan)
    df.dropna(subset=feat_cols,inplace=True)
    print(df[df[feat_cols].isna().any(axis=1)])
    
    return df


if __name__ == "__main__": 
    
    df = pd.read_csv("training.csv")
    df = split_data(df)
    x = pd.concat([df['swstr%'], df.loc[:,'stuff+':'cstr%']],axis=1)
    k = df['k%']
    bb= df['bb%']
    print(x.columns)
    print(x.columns[x.isna().any()].tolist())
    print(x.shape)
    #std_sc = StandardScaler().fit(X)
    #X = std_sc.transform(X)
    
    # Init PCA object
    pca = PCA()
    # Fit the PCA to your data
    pca.fit(x)
    # Plot cumulative explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    
    # Dictionary to hold models for each target variable
    models = {}
################################ XGB REGRESSION CODE ##############################################################
    # Initialize XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')

    # Define parameter grid
    param_grid = {
        'max_depth': [3, 6, 9],  # Maximum depth of each tree
        'learning_rate': [0.01, 0.1, 0.3],  # Learning rate
        'n_estimators': [50, 100, 200],  # Number of boosting rounds
        'colsample_bytree': [0.6, 0.8, 1.0],  # Subsample ratio of columns when constructing each tree
        'gamma': [0, 0.1, 0.3],  # Minimum loss reduction required to make a further partition on a leaf node of the tree
        'reg_alpha': [0, 0.1, 0.3],  # L1 regularization term on weights
        'reg_lambda': [0, 0.1, 0.3]  # L2 regularization term on weights
    }


    # List of target variable column names
    target_columns = [k.name, bb.name]
    feature_names = x.columns.tolist()
    # Iterate over target columns
    for target_column in target_columns:
        # Initialize GridSearchCV
        model_gridsearch = GridSearchCV(
            estimator=xgb_regressor,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,  # 5-fold cross-validation
            n_jobs=-1  # Use all available CPU cores
        )
        
        # Fit the grid search to the data for the current target column
        model_gridsearch.fit(x, df[target_column])
        
        # Store the best model in the dictionary
        models['xgb_'+target_column] = model_gridsearch.best_estimator_

        # Print the best parameters found
        print(f"Best Parameters for {target_column}:", model_gridsearch.best_params_)

        # Get feature importances for the best model
        best_model = models['xgb_'+target_column]
        feats_value = best_model.feature_importances_

        # Get the indices of features sorted by importance
        feats = np.argsort(feats_value)[::-1]

        print(f"Feature ranking for {target_column}:")
        for i, idx in enumerate(feats):
            print(f"{i + 1}. Feature {feature_names[idx]}: {feats_value[idx]}")

        # Plot feature importance for the best model
        '''
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance for {target_column}")
        plt.bar(range(len(feature_names)), feats_value[feats], color="r", align="center")
        plt.xticks(range(len(feature_names)), [feature_names[idx] for idx in feats], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Feature Importance")
        plt.show()
        '''
###########################################################################################################
    
    # Fit models for both 'k%' and 'bb%' target variables
    models['k%_poly'] = polynomial_learning(x, k, k.name)
    models['bb%_poly'] = polynomial_learning(x, bb, bb.name)   
    
    df_test = pd.read_csv("test.csv")
    df_test = split_data(df_test)
    x_test = pd.concat([df_test['swstr%'], df_test.loc[:,'stuff+':'cstr%']],axis=1)
    k_test= df_test['k%']
    bb_test = df_test['bb%']
    
    predictions_df = pd.DataFrame(index=x_test.index)
    
    for name,model in models.items():
        if 'k%' in name:
            test_data = k_test
        elif 'bb%' in name:
            test_data = bb_test
        else:
            continue  # Skip models not related to 'k%' or 'bb%'
        y_preds = evaluate_model(model, x_test, test_data, name)
        predictions_df[name] = y_preds

    # Write the DataFrame to a CSV file
    predictions_df.to_csv('pred_k_bb%.csv')
    
    players_2021 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2021].unique()
    players_2022 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2022].unique()
    common_players = set(players_2021).intersection(players_2022)
    
    
    players_2023 = df.index.get_level_values('name')[df.index.get_level_values('season') == 2023].unique()
    common_players_2 = set(players_2022).intersection(players_2023)
    print(common_players_2)
    data_2021_x = df.loc[(2021, list(common_players)), :]
    data_2022_y = df.loc[(2022, list(common_players)), :]
    data_2022_x = df.loc[(2022, list(common_players_2)), :]
    data_2023_y = df.loc[(2023, list(common_players_2)), :]
    df_x = pd.concat([data_2021_x, data_2022_x])
    df_y = pd.concat([data_2022_y, data_2023_y])
    
    print(df_x,df_y)
    x = pd.concat([df_x['swstr%'], df_x.loc[:,'stuff+':'cstr%']],axis=1)
    k_yty = df_y['k%']
    bb_yty = df_y['bb%']
    
    #k_x = models['k%_poly'].named_steps['poly'].transform(x)
    #preds_k = models['k%_poly'].named_steps['regressor'].predict(k_x)
    preds_k = models['xgb_k%'].predict(x)
    r2_kk = r2_score(k_yty,preds_k)
    
    #bb_x = models['bb%_poly'].named_steps['poly'].transform(x)
    #preds_bb = models['bb%_poly'].named_steps['regressor'].predict(bb_x)
    preds_bb = models['xgb_bb%'].predict(x)
    r2_bb = r2_score(bb_yty,preds_bb)
    
    print("K% R2 YEAR TO YEAR: ", r2_kk)
    print("BB% R2 YEAR TO YEAR: ", r2_bb)
    # # Evaluate the models
    # mse_results = {}
    # print(y_test)
    # print(predictions)
    # for name, model in models.items():
    #     mse = mean_squared_error(y_test[name.split('_')[1]], predictions[name])
    #     mse_results[name] = mse
    #     print(f"Mean Squared Error ({name}): {mse}")
    
        
    # perc_err = {}
    # for name,model in models.items():
    #     perc_error = abs((y_test[name.split('_')[1]] - predictions[name]) / y_test[name.split('_')[1]]) * 100
    #     # Calculate average percentage error
    #     avg = perc_error.mean()
    #     perc_err[name] = avg
    #     print(f"Average Percentage Error ({name}): {avg}")
    
        
    # # Choose the model with the lowest error percentage for each output
    # best_models = {}
    # for column in y_train.columns:
    #     best_model_name = min((name for name in perc_err.keys() if name.startswith(('RandomForest', 'Polynomial')) and name.endswith(column)), key=perc_err.get)
    #     best_models[column] = models[best_model_name]
    
    # #Print the best model for each output
    # print("\nBest Models:")
    # for column, model in best_models.items():
    #     print(f"Best Model for {column}: {model}")

        
    ############## RANDOM FOREST ##############################
    # for column in y.columns:
    #     test_preds[column] = models[column].predict(X_test)
    #     mse[column] = mean_squared_error(y[column],test_preds[column])
    #     tmp_perc = abs(( y[column] - test_preds[column]) / y[column]) * 100
    #     percentage_error[column] = tmp_perc.mean()
    #     r2[column] = r2_score(y[column],test_preds[column])
        
    
    
