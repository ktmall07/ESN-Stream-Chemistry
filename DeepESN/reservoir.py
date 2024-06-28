import time
import numpy as np
import reservoirpy as respy
from reservoirpy.observables import rmse, rsquare
import pandas as pd
import hydroeval as he
from matplotlib import pyplot as plt

def normal_w(n, m, **kwargs):
    return np.random.normal(0,1,size=(n,m))

NUM_MODELS = 10

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

LONGTERM_FEATURES = ['Temperature (Mean)', 'datetime']

temp_data = pd.DataFrame(pd.read_csv('colorado_river_longterm.csv'))
do_data = pd.DataFrame(pd.read_csv('colorado_river_oxygen_combined.csv'))

temp_data = temp_data[LONGTERM_FEATURES]

data = temp_data['Temperature (Mean)'].dropna().to_numpy()

temp = data.reshape(-1,1)
# discharge = data[:, 1].reshape(-1,1)
# conductance = data[:, 2].reshape(-1,1)

do_data = do_data[['Dissolved Oxygen (Mean)', 'datetime']]
do = do_data['Dissolved Oxygen (Mean)'].dropna().to_numpy()
do = do.reshape(-1, 1)

# Lists of Results from each model
temperature_rmses = []
do_rmses = []
temperature_rsquares = []
do_rsquares = []
temperature_nses = []
do_nses = []

times = []


for i in range(NUM_MODELS):
    reservoir1 = respy.nodes.Reservoir(units=1000, lr=0.9)
    readout1 = respy.nodes.Ridge(ridge=1e-7)

    reservoir2 = respy.nodes.Reservoir(units=1000, lr=0.9)
    readout2 = respy.nodes.Ridge(ridge=1e-7)

    temp_esn = reservoir1 >> readout1
    do_esn = reservoir2 >> readout2

    start = time.time()
    temp_predictions = temp_esn.fit(temp[:9000], temp[1:9001])
    temp_predictions = temp_esn.run(temp[9001:-1]) 
    end = time.time()
    do_esn.fit(do[:2300], do[1:2301])
    do_predictions = do_esn.run(do[2301:-1])

    t = end - start
    times.append(t)

    # Results
    
    # RMSE
    temp_rmse = rmse(temp[9002:], temp_predictions)
    do_rmse = rmse(do[2302:], do_predictions)
    # R2
    temp_rsquare = rsquare(temp[9002:], temp_predictions)
    do_rsquare = rsquare(do[2302:], do_predictions)
    
    # NSE
    temp_nse = he.evaluator(he.nse, list(temp_predictions), list(temp[9002:]))[0]
    do_nse = he.evaluator(he.nse, list(do_predictions), list(do[2302:]))[0]

    # Append to Metrics Lists
    temperature_rmses.append(temp_rmse)
    do_rmses.append(do_rmse)

    temperature_rsquares.append(do_rsquare)
    do_rsquares.append(do_rsquare)
    
    temperature_nses.append(temp_nse)
    do_nses.append(do_nse)

    temp_data['datetime'] = pd.to_datetime(temp_data['datetime'])
    do_data['datetime'] = pd.to_datetime(do_data['datetime'])

    # Temperature results
    plt.figure(figsize=(10, 3))
    plt.title("Temperature Model Fit")
    plt.ylabel("$Water\ Temperature\ (°C)$")
    plt.xlabel("$Time\  (Days)$")
    plt.plot(temp_data['datetime'][-1709:], temp_predictions, label='Predicted Temperature')
    plt.plot(temp_data['datetime'][-1709:], temp[9002:], label='Actual Temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figs/temperature_predictions' + str(i) + '.png')

    # Dissolved do Results
    plt.figure(figsize=(10, 3))
    plt.title("Dissolved Oxygen Model Fit")
    plt.ylabel("$Dissolved\ Oxygen\ (mg/L)$")
    plt.xlabel("$Time\ (Days)$")
    plt.plot(do_predictions, label='Predicted Dissolved Oxygen')
    plt.plot(do[2302:], label='Actual Disolved Oxygen')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figs/do_predictions' + str(i) + '.png')

    # Close Generated Figures after saving
    plt.close()

lstm_temp_fitted = pd.DataFrame(pd.read_csv('temperature_fitted.csv'))
lstm_temp_time = lstm_temp_fitted['training_time'][0]
lstm_dop_fitted = pd.DataFrame(pd.read_csv('dop_fitted.csv'))
lstm_dop_time = lstm_dop_fitted['training_time'][0]
lstm_do_fitted = pd.DataFrame(pd.read_csv('do_fitted.csv'))
lstm_do_time = lstm_do_fitted['training_time'][0]

lstm_temp_results = pd.DataFrame(pd.read_excel('D:\Machine Learning\Machine-Learning-Watersheds\DeepESN\Temperature_Results.xlsx'))
lstm_dop_results = pd.DataFrame(pd.read_excel('D:\Machine Learning\Machine-Learning-Watersheds\DeepESN\DO_Permutation_Results.xlsx'))
lstm_do_results = pd.DataFrame(pd.read_excel('D:\Machine Learning\Machine-Learning-Watersheds\DeepESN\DO_Results.xlsx'))

lstm_temp_rmse = lstm_temp_results['TestSetRMSE']
lstm_temp_rsquare = lstm_temp_results['TestSetR2']
lstm_temp_nse = he.evaluator(he.nse, lstm_temp_fitted['Actuals'][-1000:], lstm_temp_fitted['FittedVals'][-1000:])[0]

lstm_do_permutations_rmse = lstm_dop_results['TestSetRMSE']
lstm_do_permutations_rsquare = lstm_dop_results['TestSetR2']
lstm_do_permutations_nse = he.evaluator(he.nse, lstm_dop_fitted['Actuals'][-300:], lstm_dop_fitted['FittedVals'][-300:])[0]

lstm_do_rmse = lstm_do_results['TestSetRMSE']
lstm_do_rsquare = lstm_do_results['TestSetR2']
lstm_do_nse = he.evaluator(he.nse, lstm_do_fitted['Actuals'][-100:], lstm_do_fitted['FittedVals'][-100:])[0]

fig, axs = plt.subplots(1, 2, figsize=(6,3), sharey=False, sharex= True)

axs[0].boxplot(temperature_nses)
axs[0].set_xlabel('Temperature')
axs[0].set_ylim(0.0, 1.0)
axs[0].set_xticks(ticks=[])

axs[1].boxplot(do_nses)
axs[1].set_xlabel('Dissolved Oxygen')
axs[1].set_ylim(0.0, 1.0)
axs[1].set_xticks(ticks=[])
fig.suptitle('ESN Model NSE Distributions')
# plt.xlabel('Nash-Sutcliffe Efficiency')
plt.tight_layout()
plt.savefig('boxplots.png')

# Temp Figure
fig, axs = plt.subplots(1, 2, figsize=(6,3), sharey=False, sharex=True)

axs[0].bar(['ESN', 'LSTM'], [max(temperature_nses), lstm_temp_nse])
axs[0].set_ylabel('Temperature Model NSE Values')
axs[0].set_ylim(0.0, 1.0)

axs[1].bar(['ESN', 'LSTM'], [max(times), lstm_temp_time])
axs[1].set_ylabel('Training time (s)')
fig.suptitle('ESN vs LSTM Model fit and Training Time')
plt.tight_layout()
plt.savefig('temp_comparison.png')


# DO Figures
fig, axs = plt.subplots(1, 2, figsize=(6,3), sharey=False, sharex=True)

axs[0].bar(['ESN', 'LSTM'], [max(do_nses), lstm_do_permutations_nse])
axs[0].set_ylabel('Dissolved Oxygen Model NSE Values')
axs[0].set_ylim(0.0, 1.0)

axs[1].bar(['ESN', 'LSTM'], [max(times), lstm_dop_time])
axs[1].set_ylabel('Training time (s)')
fig.suptitle('ESN vs LSTM Model Fit and Training Time')
plt.tight_layout()
plt.savefig('dop_comparison.png')