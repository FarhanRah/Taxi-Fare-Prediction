import pandas
import torch
import torch.nn as nn
import numpy as np
from TabularModel import TabularModel
import time


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

# Set up the model
model = TabularModel(embed_size, conts.shape[1], 1, [200, 100], p=0.4)

# Set the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split the data in to train and test
batch_size = 60000
test_size = int(batch_size * 0.2)

cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

# Train the data
epochs = 300
losses = []
start_time = time.time()

for i in range(epochs):
    i += 1
    y_pred = model.forward(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train))  # Root Mean Square Error (RMSE)
    losses.append(loss)

    if i % 25 == 1:
        print(f'Epoch: {i:3}  Loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Duration: {(time.time() - start_time):.0f} seconds.')

# Test the data
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))

print(f'Root Mean Square Error: {loss:.8f}')

# Making sure to save the model only after the training has happened
if len(losses) == epochs:
    torch.save(model.state_dict(), 'TaxiFareRegressionModel.pt')
else:
    print('Model has not been trained. Consider loading a trained model instead.')
