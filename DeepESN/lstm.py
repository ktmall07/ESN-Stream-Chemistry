import pdb
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

LONGTERM_FEATURES = ['Temperature (Mean)', 'Discharge (Mean)', 'Specific Conductance (Mean)']

df_longterm = pd.DataFrame(pd.read_csv('colorado_river_longterm.csv'))
df_shortterm = pd.DataFrame(pd.read_csv('colorado_river_oxygen_combined.csv'))

data = df_longterm[LONGTERM_FEATURES].dropna().to_numpy()

temp = data[:, 0]

data = df_shortterm['Dissolved Oxygen (Mean)'].dropna().to_numpy()
oxygen = data

forecaster = Forecaster()