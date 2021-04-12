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

# Separate data into categorical or continuous columns
cat_cols = ['hour', 'am_or_pm', 'weekday']
cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount']

for cat in cat_cols:
    df[cat] = df[cat].astype('category')

# Merge each of the types of columns and convert them into tensors
hr = df['hour'].cat.codes.values
ampm = df['am_or_pm'].cat.codes.values
wkdy = df['weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkdy], axis=1)
cats = torch.tensor(cats, dtype=torch.int64)

conts = np.stack([df[cont].values for cont in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float)

y = torch.tensor(df[y_col].values, dtype=torch.float)

# Set up the embedding size
cats_size = [len(df[cat].cat.categories) for cat in cat_cols]
embed_size = [(size, min(50, (size+1)//2)) for size in cats_size]


