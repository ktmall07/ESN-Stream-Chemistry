import time
import numpy as np
import reservoirpy as respy
from reservoirpy import datasets
from reservoirpy.observables import rmse, rsquare
import pandas as pd
import hydroeval as he
from matplotlib import pyplot as plt

from scalecast.Forecaster import Forecaster

NUM_MODELS = 1

TEMP_PREDICTION = 'temperature_predictions'
TEMP_ACTUAL = 'temperature_actual'

ALL_FEATURES = ['Temperature (Max)', 'Temperature (Mean)', 'Temperature (Min)', 'Discharge (Mean)', 
            'Dissolved Oxygen (Max)', 'Dissolved Oxygen (Mean)', 'Dissolved Oxygen (Min)', 
            'Specific Conductance (Max)', 'Specific Conductance (Mean)', 'Specific Conductance (Min)', 
            'pH (Max)', 'pH (Median)', 'pH (Min)',
            'Turbidity (Max)', 'Turbidity (Median)', 'Turbidity (Min)']
FEATURES = ['Temperature (Mean)', 'Discharge (Mean)', 'Dissolved Oxygen (Mean)',
            'Specific Conductance (Mean)', 'pH (Median)', 'Turbidity (Median)']
ALL_LONGTERM_FEATURES = ['Temperature (Max)', 'Temperature (Mean)', 'Temperature (Min)', 'Discharge (Mean)',
                    'Specific Conductance (Max)', 'Specific Conductance (Mean)', 'Specific Conductance (Min)']
DO_FEATURES = ['Dissolved Oxygen (Mean)', 'datetime']

LONGTERM_FEATURES = ['Temperature (Mean)', 'datetime']

df_longterm = pd.DataFrame(pd.read_csv('./colorado_river_longterm.csv'))
oxygen_permutations = pd.DataFrame(pd.read_csv('./colorado_river_oxygen_combined.csv'))
oxygen = pd.DataFrame(pd.read_csv('./colorado_river_oxygen.csv'))

print(len(oxygen))
print(len(oxygen_permutations))

data = df_longterm[LONGTERM_FEATURES].dropna().to_numpy()

temp = data[:, 0]
temp_dates = data[:, 1]

dop = oxygen_permutations[['Dissolved Oxygen (Mean)', 'datetime']].dropna().to_numpy()
do = oxygen[['Dissolved Oxygen (Mean)', 'datetime']].dropna().to_numpy()

dop_do, dop_dates = dop[:, 0], dop[:, 1]
do_do, do_dates = do[:, 0], do[:, 1]



temp_forecaster = Forecaster(y=temp, test_length=1000, current_dates=temp_dates, cis=True)
dop_forecaster = Forecaster(y=dop_do, test_length=300, current_dates=dop_dates, cis=True)
do_forecaster = Forecaster(y=do_do, test_length=100, current_dates=do_dates, cis=True)

temp_forecaster.set_test_length(1000)
temp_forecaster.generate_future_dates(1000)
temp_forecaster.set_estimator('lstm')

dop_forecaster.set_test_length(300)
dop_forecaster.generate_future_dates(300)
dop_forecaster.set_estimator('lstm')

do_forecaster.set_test_length(100)
do_forecaster.generate_future_dates(100)
do_forecaster.set_estimator('lstm')

start = time.time()
temp_forecaster.manual_forecast(call_me='Temperature', lags= 100, epochs= 50)
end = time.time()
temp_forecaster.plot_test_set(ci=True)
plt.ylabel('Temperature (°C)')

temp_time = end - start
print('Temperature Time: ', temp_time) 

start = time.time()
dop_forecaster.manual_forecast(call_me='DO With Permutations', lags=20, epochs=50)
end = time.time()
dop_forecaster.plot_test_set(ci=True)
plt.ylabel('Dissolved Oxygen (mg/L)')



dop_time = end - start
print('DO with Permutations Time: ', dop_time)

start = time.time()
do_forecaster.manual_forecast(call_me='DO', lags=10, epochs=50)
end = time.time()
do_forecaster.plot_test_set(ci=True)


do_time = end - start
print('DO time: ', do_time)

temp_forecaster.export('model_summaries', determine_best_by='TestSetR2', to_excel=True, excel_name='Temperature_Results.xlsx')[['ModelNickname', 'TestSetRMSE', 'TestSetR2']]
dop_forecaster.export('model_summaries', determine_best_by='TestSetR2', to_excel=True, excel_name='DO_Permutation_Results.xlsx')[['ModelNickname', 'TestSetRMSE', 'TestSetR2']]
do_forecaster.export('model_summaries', determine_best_by='TestSetR2', to_excel=True, excel_name='DO_Results.xlsx')[['ModelNickname', 'TestSetRMSE', 'TestSetR2']]

fitted_temp = temp_forecaster.export_fitted_vals('Temperature')
fitted_temp['training_time'] = np.zeros(len(fitted_temp))
fitted_temp['training_time'][0] = temp_time
fitted_dop = dop_forecaster.export_fitted_vals('DO With Permutations')
fitted_dop['training_time'] = np.zeros(len(fitted_dop))
fitted_dop['training_time'][0] = dop_time
fitted_do = do_forecaster.export_fitted_vals('DO')
fitted_do['training_time'] = np.zeros(len(fitted_do))
fitted_do['training_time'][0] = do_time

fitted_temp.to_csv('temperature_fitted.csv')
fitted_dop.to_csv('dop_fitted.csv')
fitted_do.to_csv('do_fitted.csv')

plt.show()