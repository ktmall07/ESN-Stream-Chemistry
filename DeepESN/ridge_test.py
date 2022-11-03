import pdb
import numpy as np
import reservoirpy as respy
from reservoirpy.observables import rmse, rsquare
import pandas as pd
import hydroeval as he
from matplotlib import pyplot as plt

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

data = df_longterm[LONGTERM_FEATURES].dropna().to_numpy()

temp = data.reshape(-1,1)

res1 = respy.nodes.Reservoir(units=1000, lr = 0.9)
res2 = respy.nodes.Reservoir(units=1000, lr=0.9)
res3 = respy.nodes.Reservoir(units=1000, lr=0.9)

read1 = respy.nodes.Ridge()
read2 = respy.nodes.Ridge(ridge=1e-7)
read3 = respy.nodes.Ridge(ridge=1e-15)

small = res1 >> read1
med = res2 >> read2
large = res3 >> read3

sp = small.fit(temp[:9000], temp[1:9001])
mp = med.fit(temp[:9000], temp[1:9001])
lp = large.fit(temp[:9000], temp[1:9001])
small_predictions = small.run(temp[9001:-1])
med_predictions = med.run(temp[9001:-1])
large_predictions = large.run(temp[9001:-1])

small_rmse = rmse(temp[9002:], small_predictions)
small_rsquare = rsquare(temp[9002:], small_predictions)
small_nse = he.evaluator(he.nse, list(small_predictions), list(temp[9002:]))[0]

med_rmse = rmse(temp[9002:], med_predictions)
med_rsquare = rsquare(temp[9002:], med_predictions)
med_nse = he.evaluator(he.nse, list(med_predictions), list(temp[9002:]))[0]

large_rmse = rmse(temp[9002:], large_predictions)
large_rsquare = rsquare(temp[9002:], large_predictions)
large_nse = he.evaluator(he.nse, list(large_predictions), list(temp[9002:]))[0]

plt.figure(figsize=(10, 3))
plt.title("Temperature Predictions.")
plt.ylabel("$Temperature(t) (C)$")
plt.xlabel("$t$")
plt.plot(small_predictions, label='ridge=0')
plt.plot(med_predictions, label='ridge=1e-7')
plt.plot(large_predictions, label='ridge=1e-15')
plt.plot(temp[9002:], label='Actual')
plt.legend()
plt.savefig('ridge_size.png')
