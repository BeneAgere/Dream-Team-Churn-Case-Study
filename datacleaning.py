import pandas as pd
from datetime import datetime, timedelta


def dataClean(filename):
    df = pd.read_csv('churn.csv',parse_dates=['last_trip_date','signup_date'])
    avg_by_driver = df['avg_rating_by_driver'].sum()*1.0/len(df['avg_rating_by_driver'])
    df['avg_rating_by_driver'] = df['avg_rating_by_driver'].fillna(avg_by_driver)
    avg_of_driver = df['avg_rating_of_driver'].sum()*1.0/len(df['avg_rating_of_driver'])
    df['avg_rating_of_driver'] = df['avg_rating_of_driver'].fillna(avg_of_driver)
    df['phone'] = df['phone'].fillna('others')
    df = pd.concat((df, pd.get_dummies(df['phone']), pd.get_dummies(df['city']),pd.get_dummies(df['luxury_car_user'])), axis=1)
    df['churn'] = datetime(2014, 7, 1) - df['last_trip_date'] > timedelta(days=30)
    df = df.drop(['luxury_car_user', 'phone', 'city'], axis=1)
    return df
