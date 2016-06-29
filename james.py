import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn-metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScale
from sklearn.neighbors import KNeighborsClassifier


def dataClean(filename):
    df = pd.read_csv('churn.csv',parse_dates=['last_trip_date','signup_date'])
    avg_by_driver = df['avg_rating_by_driver'].sum()*1.0/len(df['avg_rating_by_driver'])
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(avg_by_driver)
    avg_of_driver = df['avg_rating_of_driver'].sum()*1.0/len(df['avg_rating_of_driver'])
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(avg_of_driver)
    df['phone'] = df['phone'].fillna('others')
    df = pd.concat((df, pd.get_dummies(df['phone']), pd.get_dummies(df['city']),pd.get_dummies(df['luxury_car_user'])), axis=1)
    df['churn'] = datetime(2014, 7, 1) - df['last_trip_date'] > timedelta(days=30)
    df['days_of_signed'] = (datetime(2014,7,1) - df['signup_date'])
    df['days_of_signed'] = df['days_of_signed'].apply(lambda x: x.days)
    df = df.drop(['luxury_car_user', 'phone', 'city'], axis=1)
    return df

def confusion_matrix(y_true, y_predict):
    tp = np.sum(np.logical_and(y_true, y_predict))
    fp = np.sum(np.logical_and(np.logical_xor(y_true, y_predict), y_predict))
    fn = np.sum(np.logical_and(np.logical_xor(y_true, y_predict), y_true))
    tn = np.sum(np.logical_not(np.logical_or(y_true,  y_predict)))
    return np.array([[tp, fp],[fn, tn]])


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


if __name__ == '__main__':
   df = dataClean()
   y = df['churn']
   X = df.drop('churn')
   X_scaled, y_scaled = StandardScale().fit_transform(X, y)
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)
   # print decisionTree(X_train, X_test, y_train, y_test) (edited)

   plt.show()
