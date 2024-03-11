# import starting packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc, recall_score

# import the training data
data_raw = pd.read_excel('~/Desktop/master-thesis-code/data.xlsx', index_col=0)

# set up a params grid to find the best performing model
# we can pass in each of the models
# there is no need to redfine this for each model
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced']
}

# set up function to run the the random forest
def run_random_forest(data, lag, test_size, scoring, params):
    
    # initiate the model
    rf_classifier = RandomForestClassifier(random_state=42)
    
    # modify dataset for lag
    data[f"nber_recession_{lag}_month_lag"] = data['nber_recession'].shift(lag)
    
    # drop the original recession column and na values
    data = data.drop(columns=['nber_recession'])
    data = data.dropna()
    
    # set up training and testing data
    X = data.drop(columns=[f"nber_recession_{lag}_month_lag"])
    y = data[f"nber_recession_{lag}_month_lag"]
    
    # split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # set up the grid search object to perform the analysis
    # set cross validation to 5 which is a standard benchmark
    grid_search_cv = GridSearchCV(estimator=rf_classifier, param_grid=params, cv=5, scoring=scoring)
    
    # perform the initial grid search
    grid_search_cv.fit(X_train, y_train)
    
    # get the best performing model
    best_parameters = grid_search_cv.best_params_
    best_model = grid_search_cv.best_estimator_
    
    # predict the results
    y_pred = best_model.predict(X_test)
    
    # create a confusion matrix to visualize results
    conf_mat = confusion_matrix(y_test, y_pred)

    # calculate precision score for testing data
    precision = precision_score(y_test, y_pred)

    # calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # calculate recall score
    recall = recall_score(y_test, y_pred)
    
    # calculate the f1 score
    f1 = f1_score(y_test, y_pred)

    # calculate auc_roc score
    auc_roc = roc_auc_score(y_test, y_pred)
    
    return {'best_parameters': best_parameters,
            'best_model': best_model,
            'confusion_matrix': conf_mat,
            'precision_score': precision,
            'accuracy_score': accuracy,
            'recall_score': recall,
            'f1_score': f1,
            'auc_roc_score': auc_roc}

# set up lag list
#lags = [3, 6, 9, 12, 18]

lags = [3,6]

# make a list of the resulting ranom forest
results = [(f"{lag}_month_lag_results", run_random_forest(data_raw, lag, 0.2, 'precision', param_grid)) for lag in lags]










