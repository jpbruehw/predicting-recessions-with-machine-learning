# import libraries
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc, recall_score

# import the training data
data_raw = pd.read_excel('~/Desktop/master-thesis-code/data.xlsx', index_col=0)

# set up lags for looping
lags = [3, 6, 9, 12, 18]

# function to perform analysis
def run_neural_network(data, lag, test_size, epochs, batch_size, validation_split, optimizer, loss, metrics):
    """
    Function to run neural network based on number of lags
    each lag has its own model
    """
    
    # make a copy of the original DataFrame to avoid modifying it
    data_copy = data.copy()
    
    # modify dataset for lag
    # we want to set the recession indicator back by the lag so that t0 is aligned with t+lag
    data_copy[f"nber_recession_{lag}_month_lag"] = data_copy['nber_recession'].shift(-lag)
    
    # drop the original recession column and na values
    data_copy = data_copy.drop(columns=['nber_recession'])
    data_copy = data_copy.dropna()
    
    # set up training and testing data
    X = data_copy.drop(columns=[f"nber_recession_{lag}_month_lag"])
    y = data_copy[f"nber_recession_{lag}_month_lag"]
    
    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # build initial model
    # two layer feed-forward neural network
    # pass in 1 to ouput layer since we have one class
    model = Sequential([
        Dense(64, activation='relu',
                  input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')])
    
    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)
    
    # test the data and extract the metrics
    results = model.evaluate(X_test, y_test)
    
    # return the data
    return {'data': data,
            'model': model,
            'results': results,
            'history': history}

# run model for different lags
# returns tuple of the lag and the results
neural_network_results = [(f"{lag}_month_lag_results", run_neural_network(data_raw, lag, 0.2, 10, 32, 0.2,
                        'adam', 'cross_entropy', ['precision_score', 'f1_score',
                        'accuracy_score', 'roc_auc_score', 'recall_score'])) for lag in lags]
