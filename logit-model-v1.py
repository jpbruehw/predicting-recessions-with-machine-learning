# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score, roc_auc_score, recall_score

# import the training data
data_raw = pd.read_excel('~/Desktop/master-thesis-code/data.xlsx', index_col=0)

# set up lags for looping
lags = [3, 6, 9, 12, 18]

# set the parameter grid
param_grid = {
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200, 300, 400, 500],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

# set up function to handle the logit model
def run_logit_regression(data, lag, test_size, scoring, params, stratify):
    
    """
    This function takes various inputs 
    and returns summary statistics
    for the logistic regression model
    some inputs persist across iterations
    """
    
    # make a copy of the original DataFrame to avoid modifying it
    data_copy = data.copy()
    
    # initiliaze the logistic regression model
    logistic_regression = LogisticRegression(random_state=42, verbose=1, n_jobs=12)
    
    # modify dataset for lag
    # we want to set the recession indicator back by the lag so that t0 is aligned with t+lag
    data_copy[f"nber_recession_{lag}_month_lag"] = data_copy['nber_recession'].shift(-lag)
    
    # drop the original recession column and na values
    data_copy = data_copy.drop(columns=['nber_recession'])
    data_copy = data_copy.dropna()
    
    # set up training and testing data
    X = data_copy.drop(columns=[f"nber_recession_{lag}_month_lag"])
    y = data_copy[f"nber_recession_{lag}_month_lag"]
    
    # calculate the class weights manually
    class_counts = y.value_counts()
    total_samples = len(y)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    
    # update the parameters grid
    params['class_weight'] = [None, 'balanced', class_weights]

    # split data into training and test data
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
    # apply SMOTE only to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # set up the grid search object to perform the analysis
    # set cross validation to 5 which is a standard benchmark
    grid_search_cv = GridSearchCV(estimator=logistic_regression, param_grid=params, cv=5, scoring=scoring)
    
    # perform the initial grid search
    grid_search_cv.fit(X_train_resampled, y_train_resampled)
    
    # get the best performing model
    best_parameters = grid_search_cv.best_params_
    best_model = grid_search_cv.best_estimator_
    
    # predict the results
    y_pred = best_model.predict(X_test)
    
    # create a confusion matrix to visualize results
    conf_mat = confusion_matrix(y_test, y_pred)

    # get predicted values and metrics
    metrics_obj= {
       'accuracy': accuracy_score(y_test, y_pred),
       'precision': precision_score(y_test, y_pred),
       'recall': recall_score(y_test, y_pred),
       'f1': f1_score(y_test, y_pred),
       'roc_auc': roc_auc_score(y_test, y_pred),
       }

    
    return {'data': data_copy,
            'class_weights': class_weights,
            'best_parameters': best_parameters,
            'best_model': best_model,
            'y_true': y_test,
            'y_pred': y_pred,
            'confusion_matrix': conf_mat,
            'model_metrics': metrics_obj}

# make a list of the resulting logistic models
logit_results = [(f"{lag}_month_lag_results", run_logit_regression(data_raw, lag, 0.2, 'precision', param_grid, False)) for lag in lags]

# make a dataframe of all accuracy results
headers_metrics = ['lag', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# store the results for each iteration
iteration_metrics = []
# iterate over results
for result in logit_results:
    # extract from the tuple
    metrics = result[1]['model_metrics']
    # extract each value
    values = [val for _, val in metrics.items()]
    # insert name of lag
    values.insert(0, result[0])
    # append to the list
    iteration_metrics.append(values)
# convert to a dataframe
metric_data = pd.DataFrame(iteration_metrics, columns=headers_metrics)

print(metric_data)
