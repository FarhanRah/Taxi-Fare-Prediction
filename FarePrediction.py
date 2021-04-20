import torch
import torch.nn as nn
import numpy as np
import pandas
from TabularModel import TabularModel


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


def user_data(model):
    # INPUT NEW DATA
    plat = float(input('What is the pickup latitude?  '))
    plong = float(input('What is the pickup longitude? '))
    dlat = float(input('What is the dropoff latitude?  '))
    dlong = float(input('What is the dropoff longitude? '))
    psngr = int(input('How many passengers? '))
    dt = input('What is the pickup date and time?\nFormat as YYYY-MM-DD HH:MM:SS     ')

    # PREPROCESS THE DATA
    dfx_dict = {'pickup_latitude': plat, 'pickup_longitude': plong, 'dropoff_latitude': dlat,
                'dropoff_longitude': dlong, 'passenger_count': psngr, 'EDTdate': dt}
    dfx = pandas.DataFrame(dfx_dict, index=[0])
    dfx['dist_km'] = haversine_distance(dfx, 'pickup_latitude', 'pickup_longitude',
                                        'dropoff_latitude', 'dropoff_longitude')
    dfx['EDTdate'] = pandas.to_datetime(dfx['EDTdate'])

    # We can skip the .astype(category) step since our fields are small,
    # and encode them right away
    dfx['Hour'] = dfx['EDTdate'].dt.hour
    dfx['AMorPM'] = np.where(dfx['Hour'] < 12, 0, 1)
    dfx['Weekday'] = dfx['EDTdate'].dt.strftime("%a")
    dfx['Weekday'] = dfx['Weekday'].replace(['Fri', 'Mon', 'Sat', 'Sun', 'Thu', 'Tue', 'Wed'],
                                            [0, 1, 2, 3, 4, 5, 6]).astype('int64')
    # CREATE CAT AND CONT TENSORS
    cat_cols = ['Hour', 'AMorPM', 'Weekday']
    cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                 'dropoff_longitude', 'passenger_count', 'dist_km']
    xcats = np.stack([dfx[col].values for col in cat_cols], 1)
    xcats = torch.tensor(xcats, dtype=torch.int64)
    xconts = np.stack([dfx[col].values for col in cont_cols], 1)
    xconts = torch.tensor(xconts, dtype=torch.float)

    # PASS NEW DATA THROUGH THE MODEL WITHOUT PERFORMING A BACKPROP
    with torch.no_grad():
        z = model(xcats, xconts)
    print(f'\nThe predicted fare amount is ${(z.item() + 3):.2f}')  # We are adding $3 to consider the inflation over the past years


# Defining the model
emb_sizes = [(24, 12), (2, 1), (7, 4)]
model2 = TabularModel(emb_sizes, 6, 1, [200, 100], p=0.4)

# Loading the saved stats
model2.load_state_dict(torch.load('TaxiFareRegressionModel.pt'))
model2.eval()

user_data(model2)
