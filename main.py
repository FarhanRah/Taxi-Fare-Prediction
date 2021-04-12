import pandas
import torch
import numpy as np


# Calculates the distance in km using the longitudes and latitudes
def haversine_distance(df, lat1, long1, lat2, long2):
    r = 6371  # average radius of Earth in kilometers

    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)  # in kilometers

    return d


df = pandas.read_csv('CanadaTaxiFares.csv')

# Feature engineering the data
df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df['pickup_datetime'] = pandas.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['am_or_pm'] = np.where(df['hour'] > 12, 'pm', 'am')
df['weekday'] = df['pickup_datetime'].dt.strftime('%a')
