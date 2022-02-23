'''
Random search with negative rmse as function
'''

#imports

from TaxiFareModel.trainer import Trainer
from TaxiFareModel.data import get_data, clean_data

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

def rand_scores(X, y, mod_dict):
    res_dict = {}
    for k, v in mod_dict.items():
        model = Trainer(X, y, v)
        results = cross_val_score(model.set_pipeline(), X, y, cv=5,
                                  scoring='neg_root_mean_squared_error')
        res_dict[k] = results.mean()
    print(res_dict)
    return res_dict

if __name__ == '__main__':
    df = get_data()
    df = clean_data(df)
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']

    module_dict = {
    'SVM_regressor_rbf_0.001': SVR(C=10, gamma = 0.001, kernel='rbf'),
    'SVM_regressor_rbf_0.01': SVR(C=10, gamma = 0.01, kernel='rbf'),
    'SVM_regressor_rbf_0.1': SVR(C=10, gamma = 0.1, kernel='rbf'),
    'SVM_regressor_rbf_1': SVR(C=10, gamma = 1, kernel='rbf'),
    }

    cv_score(X, y, module_dict)
