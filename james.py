import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

def dataClean(filename):
    df = pd.read_csv(filename, parse_dates=['last_trip_date','signup_date'])
    avg_by_driver = df['avg_rating_by_driver'].sum()*1.0/len(df['avg_rating_by_driver'])
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(avg_by_driver)
    avg_of_driver = df['avg_rating_of_driver'].sum()*1.0/len(df['avg_rating_of_driver'])
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(avg_of_driver)
    df['phone'] = df['phone'].fillna('others')
    df = pd.concat((df, pd.get_dummies(df['phone']), pd.get_dummies(df['city'])), axis=1)
    df['luxury_car_user'] = df['luxury_car_user'].astype(int)
    df['churn'] = datetime(2014, 7, 1) - df['last_trip_date'] > timedelta(days=30)
    df['days_of_signed'] = (datetime(2014,7,1) - df['signup_date'])
    df['days_of_signed'] = df['days_of_signed'].apply(lambda x: x.days)
    df.rename(columns={'True': 'Luxury_car_user', 'False': 'Non_luxury_car_user'}, inplace=True)
    df = df.drop(['phone', 'city', 'signup_date', 'last_trip_date'], axis=1)
    return df

def confusion_matrix(y_true, y_predict):
    tp = np.sum(np.logical_and(y_true, y_predict))
    fp = np.sum(np.logical_and(np.logical_xor(y_true, y_predict), y_predict))
    fn = np.sum(np.logical_and(np.logical_xor(y_true, y_predict), y_true))
    tn = np.sum(np.logical_not(np.logical_or(y_true,  y_predict)))
    return np.array([[tp, fp],[fn, tn]])

def kfold_val(X_train, y_train, k, model):
    kf = KFold(len(y), k)
    accuracy = []
    precision = []
    recall = []
    mse = []
    r2 = []
    auc = []
    for train_index, val_index in kf:
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        fit = model.fit(X_train, y_train)
        y_hat = fit.predict(X_val)

        accuracy.append(accuracy_score(y_val,y_hat))
        precision.append(precision_score(y_val,y_hat))
        recall.append(recall_score(y_val,y_hat))
        mse.append(mean_squared_error(y_val, y_hat))
        r2.append(r2_score(y_val, y_hat))
        auc.append(roc_auc_score(y_val, y_hat))
    return [np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(mse),   np.mean(r2), np.mean(auc)]


def correlationPlot(df):
    pd.scatter_matrix(df, alpha=0.2, figsize=(15,15), diagonal='kde')
    plt.matshow(df.corr())

def score(y_true, y_pred):
    #Calculate AUC, MSE, adjusted r squared
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return mse, r2, auc

def KNN(X, y):
    knnMod = KNeighborsClassifier()
    parms
    pass

def randomForest(X, y):
    rf = RandomForestClassifier()
    parms = {'n_estimators':[10, 50, 100, 150, 200, 300],
                'max_features':[4, 6, 8, 10],
                'criterion':['gini', 'entropy'],
                'oob_score':[True],
                'n_jobs': [-1]}
    gsMod = GridSearchCV(estimator=rf, param_grid=parms, n_jobs=-1, cv=10, verbose=2)
    gsMod.fit(X, y)
    best_parms = gsMod.best_params_
    print 'scores:', gdMod.grid_scores_
    best_score = gsMod.best_score_
    return gsMod.best_estimator_


if __name__ == '__main__':
   df = dataClean('data/churn.csv')
   y = df['churn'].values.astype(int)
   X = df.drop(['churn'], axis=1).values
   scaler = StandardScaler()
   X_scaled= scaler.fit_transform(X)
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
   # print decisionTree(X_train, X_test, y_train, y_test) (edited)
   bestRF = randomForest(X_train, y_train)
   # print X_scaled
   plt.show()
