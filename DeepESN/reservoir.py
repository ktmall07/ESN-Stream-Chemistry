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

LONGTERM_FEATURES = ['Temperature (Mean)']

df_longterm = pd.DataFrame(pd.read_csv('colorado_river_longterm.csv'))
df_shortterm = pd.DataFrame(pd.read_csv('colorado_river_oxygen_combined.csv'))

data = df_longterm[LONGTERM_FEATURES].dropna().to_numpy()

temp = data.reshape(-1,1)
# discharge = data[:, 1].reshape(-1,1)
# conductance = data[:, 2].reshape(-1,1)

data = df_shortterm['Dissolved Oxygen (Mean)'].dropna().to_numpy()
oxygen = data.reshape(-1, 1)

# Lists of Results from each model
temperature_rmses = []
oxygen_rmses = []
temperature_rsquares = []
oxygen_rsquares = []
temperature_nses = []
oxygen_nses = []

times = []


for i in range(NUM_MODELS):
    reservoir1 = respy.nodes.Reservoir(units=1000, lr=0.9)
    readout1 = respy.nodes.Ridge(ridge=1e-7)

    reservoir2 = respy.nodes.Reservoir(units=1000, lr=0.9)
    readout2 = respy.nodes.Ridge(ridge=1e-7)

    temp_esn = reservoir1 >> readout1
    oxygen_esn = reservoir2 >> readout2

    start = time.time()
    temp_predictions = temp_esn.fit(temp[:9000], temp[1:9001])
    temp_predictions = temp_esn.run(temp[9001:-1]) 
    end = time.time()
    oxygen_esn.fit(oxygen[:2300], oxygen[1:2301])
    oxygen_predictions = oxygen_esn.run(oxygen[2301:-1])

    t = start - end
    times.append(t)

    # Results
    temp_rmse = rmse(temp[9002:], temp_predictions)
    oxygen_rmse = rmse(oxygen[2302:], oxygen_predictions)
    temp_rsquare = rsquare(temp[9002:], temp_predictions)
    oxygen_rsquare = rsquare(oxygen[2302:], oxygen_predictions)
    temp_nse = he.evaluator(he.nse, list(temp_predictions), list(temp[9002:]))[0]
    oxygen_nse = he.evaluator(he.nse, list(oxygen_predictions), list(oxygen[2302:]))[0]

    temperature_rmses.append(temp_rmse)
    oxygen_rmses.append(oxygen_rmse)
    temperature_rsquares.append(oxygen_rsquare)
    oxygen_rsquares.append(oxygen_rsquare)
    temperature_nses.append(temp_nse)
    oxygen_nses.append(oxygen_nse)

    print("Temp rmse: ", temp_rmse)
    print("Temp rsquare: ", temp_rsquare)
    print("Temp nse: ", temp_nse)

    print("Oxygen rmse: ", oxygen_rmse)
    print("Oxygen rsquare", oxygen_rsquare)
    print("Oxygen_nse: ", oxygen_nse)

    # # Temperature results
    # plt.figure(figsize=(10, 3))
    # plt.title("Temperature Predictions.")
    # plt.ylabel("$Temperature(t) (C)$")
    # plt.xlabel("$t$")
    # plt.plot(temp_predictions, label='Predictions')
    # plt.plot(temp[9002:], label='Actual')
    # plt.legend()
    # # plt.show()
    # plt.savefig('temperature_predictions' + str(i) + '.png')


    # # Dissolved Oxygen Results
    # plt.figure(figsize=(10, 3))
    # plt.title("Dissolved Oxygen Predictions.")
    # plt.ylabel("$Oxygen(t)$")
    # plt.xlabel("$t$")
    # plt.plot(oxygen_predictions, label='Predictions')
    # plt.plot(oxygen[2302:], label='Actual ')
    # plt.legend()
    # # plt.show()
    # plt.savefig('oxygen_predictions' + str(i) + '.png')

# plt.figure(figsize=(10,3))
# plt.title('Water Temperature Model NSE Values')
# plt.hist(temperature_nses)
# # plt.xlabel('NSE Values')
# plt.savefig('temp_nse_hist.png')

# plt.figure(figsize=(10,3))
# plt.title('Dissolved Oxygen Model NSE Values')
# plt.hist(oxygen_nses)
# # plt.xlabel('NSE values')
# plt.tight_layout()
# plt.savefig('do_nse_hist.png')

print(temperature_nses)
print(times)