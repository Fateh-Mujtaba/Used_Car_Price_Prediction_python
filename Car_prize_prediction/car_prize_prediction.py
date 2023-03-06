import pandas as pd
import numpy as np

#%% Load The data

data_df=pd.read_csv("train-data.csv")
#%%
#%% Trimmed un wanted features
data_df.drop('New_Price', axis=1, inplace=True)
data_df.drop('Unnamed', axis=1, inplace=True)

#%% Cleaning  Training data

data_df['Mileage']=data_df['Mileage'].str.replace('kmpl','')
data_df['Mileage']=data_df['Mileage'].str.replace('km/kg','')
data_df['Engine']=data_df['Engine'].str.replace('CC','')
data_df['Power']=data_df['Power'].str.replace('bhp','')
data_df['Transmission']=data_df['Transmission'].str.replace('Manual','0')
data_df['Transmission']=data_df['Transmission'].str.replace('Automatic','1')

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
data_df['Fuel_Type']=le.fit_transform(data_df['Fuel_Type'])
data_df['Owner_Type']=le.fit_transform(data_df['Owner_Type'])
data_df['Name']=le.fit_transform(data_df['Name'])
data_df['Location']=le.fit_transform(data_df['Location'])

#%% Converting Strings (Objects) into

data_df['Mileage']=data_df['Mileage'].astype(float)
data_df['Transmission']=data_df['Transmission'].astype(float)
data_df['Engine']=data_df['Engine'].astype(float)
data_df['Power'] = pd.to_numeric(data_df['Power'], errors='coerce')
data_df['Power'] = data_df['Power'].astype(float)
#%% Removing zeros from the data
data_df = data_df.fillna(0)
arr=['Mileage','Engine','Seats']
for i in arr:
    mean_value = data_df.loc[data_df[i] != 0, i].mean()
    data_df[i] = data_df[i].replace('nan', mean_value)
    
#%% Spliting data for input and ooutput

data_x=data_df.values[:,:-1]
data_y=data_df.values[:,np.newaxis,-1]
#%% Dividing data for training and testing
def splitdata(X,y):
    r=0.3
    ln=len(X)
    ts_Ins=round(ln*r)
    tr_x = X[:-ts_Ins,:]
    ts_x = X[-ts_Ins:,:]
    tr_y = y[:-ts_Ins]
    ts_y = y[-ts_Ins:]
    return tr_x,ts_x,tr_y,ts_y

train_x,test_x,train_y,test_y=splitdata(data_x,data_y)



#%% Training the model through linear regression
#%%
from sklearn import linear_model
reg_car_price=linear_model.LinearRegression().fit(train_x,train_y)
#%% prediction
pred_y=reg_car_price.predict(test_x)
#%% Evaluation
from sklearn.metrics import mean_squared_error,r2_score
print("\nCorelation Coefficient: ",reg_car_price.coef_)
print("\nMSE: %.2f" % mean_squared_error(test_y,pred_y))
print("\nR2 Score: %.2f" % r2_score(test_y,pred_y))


