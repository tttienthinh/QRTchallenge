"""
exit()
cd data
python3
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split 


# Data
X_train = pd.read_csv('X_train.csv')
Y_train = pd.read_csv('Y_train.csv')
X_test = pd.read_csv('X_test.csv')
TEST_PROPORTION = 0.1

data = X_train.join(Y_train.set_index("ID"), on="ID")
indices = np.arange(len(data))
np.random.shuffle(indices)
splitter = int(TEST_PROPORTION * len(data))

test_data = data.iloc[indices[:splitter]].reset_index().drop("index", axis=1) 
train_data = data.iloc[indices[splitter:]].reset_index().drop("index", axis=1) 



# Functions
def cross_concat(df):
    suff = "_add"
    df_result = pd.DataFrame()
    for i in range(len(df)): # Create dataframe with same index i
        df_result_i = df.copy()
        df_result_i[[col + suff for col in df.columns]] = df.loc[i] # Get all values of line i
        df_result = pd.concat([df_result, df_result_i])
    # Create order value
    df_result["order"] = (df_result.TARGET < df_result.TARGET_add).astype(int)
    # Reset index
    return df_result.reset_index().drop("index", axis=1)  


## Saving data

cross_test = cross_concat(test_data)
cross_train = cross_concat(train_data)

cross_test.to_csv("2-ordre_test.csv")
cross_train.to_csv("2-ordre_train.csv")

cross_test = pd.read_csv("2-ordre_test.csv")
cross_train = pd.read_csv("2-ordre_train.csv")

# LigthGBM (abandon)
import lightgbm as lgb


train_data = lgb.Dataset(
    cross_train.drop(["ID", "ID_add", "TARGET", "TARGET_add", "order"], axis=1), 
    label=cross_train["order"],
    categorical_feature=["COUNTRY", "DAY_ID"]
)
test_data = lgb.Dataset(
    cross_test.drop(["ID", "ID_add", "TARGET", "TARGET_add", "order"], axis=1), 
    label=cross_train["order"],
    categorical_feature=["COUNTRY", "DAY_ID"],
    reference=train_data
)
# train_data.save_binary('train.bin')

# training
num_round = 10
param = {'num_leaves': 31, 'objective': 'binary', 'metric': 'auc'}
bst = lgb.train(param, train_data, num_round, valid_sets=test_data) 


# Catboost
from catboost import CatBoostRegressor, CatBoostClassifier
import sklearn

# Initialize model
model = CatBoostClassifier(iterations=2,
                           learning_rate=1,
                           depth=2)
model = CatBoostClassifier(iterations=10,
                           learning_rate=1,
                           depth=2,
                           custom_metric=['Logloss',
                                          'AUC:hints=skip_train~false'])
# Fit model
model.fit(
    cross_train.drop(["ID", "ID_add", "TARGET", "TARGET_add", "order"], axis=1), 
    cross_train["order"],
    eval_set=(
        cross_test.drop(["ID", "ID_add", "TARGET", "TARGET_add", "order"], axis=1), 
        cross_test["order"]
    ),
    cat_features=["COUNTRY", "DAY_ID", "COUNTRY_add", "DAY_ID_add"], 
    use_best_model=True
)
model.save_model("2-relationOrdre_CatBoost")

sklearn.metrics.accuracy_score(
    cross_test["order"],
    model.predict(cross_test.drop(["ID", "ID_add", "TARGET", "TARGET_add", "order"], axis=1))
)
