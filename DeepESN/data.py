import pandas as pd

PATH = "green_river_data.txt"

csv = pd.read_csv(PATH)

csv.to_csv(r'green_river.csv', index=None)