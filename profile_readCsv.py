import pandas as pd
import matplotlib.pyplot as plt #backend is QtAgg on Windows 10
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import statsmodels.api as sm
import sys,os

# import find_modes

from line_profiler import profile



#currently there is an import error
# ImportError: cannot import name 'profile' from 'line_profiler' (C:\Users\kyanzhe\AppData\Local\anaconda3\envs\env4\lib\site-packages\line_profiler\__init__.py)
#TODO fix


# This will suppress all warnings
warnings.filterwarnings("ignore")

@profile
def read_csv(csv, position_percent=100, smooth=False):

    df = csv
    

    # non_trimmed = df.iloc[::10] #subsample by a factor of 10
    index_position = int(len(df)* float(position_percent)/100)-1 
    trimmed_df = df.iloc[:index_position:]
    
    # Extract data from the subsampled dataframe
    x = trimmed_df['.x']
    y = trimmed_df['.y']
    z = trimmed_df['.z']
    xvel = trimmed_df['.xVelocity']
    yvel = trimmed_df['.yVelocity']
    zvel = trimmed_df['.zVelocity']
    timestamps = trimmed_df['.timestamp']  # should use the second column, .timestamp


            
    
    if smooth:
        # Apply Lowess smoothing to each dimension
        smoothing_frac = 90/len(x)  # Smoothing fraction, you can adjust this value

        smoothed_x = sm.nonparametric.lowess(x, timestamps, frac=smoothing_frac, it=0)[:, 1]
        smoothed_y = sm.nonparametric.lowess(y, timestamps, frac=smoothing_frac, it=0)[:, 1]
        smoothed_z = sm.nonparametric.lowess(z, timestamps, frac=smoothing_frac, it=0)[:, 1]

        # Create Pandas Series with smoothed data and timestamps
        smoothed_x_series = pd.Series(smoothed_x, index=timestamps)
        smoothed_y_series = pd.Series(smoothed_y, index=timestamps)
        smoothed_z_series = pd.Series(smoothed_z, index=timestamps)
        return smoothed_x_series, smoothed_y_series, smoothed_z_series, xvel, yvel, zvel, timestamps
    else:
        return x, y, z, xvel, yvel, zvel, timestamps

# #session 2
csv_path_2 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 1.csv" #-0.2 to 0.35m. this seems to be an ideal results
csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 2.csv" #-0.5 to 0.2m. this has error, beginning around 45%



# csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 2.csv" #ends around -0.8m. this seems to be better

auto_increment = False
highlight_cumulative_overlap = False

show_second_plot = True

# Load the CSV file into a DataFrame, skipping the first row
csv1 = pd.read_csv(csv_path_1, skiprows=1)
csv2 = pd.read_csv(csv_path_2, skiprows=1)


x1, y1, z1, xvel1, yvel1, zvel1, timestamp1 = read_csv(csv1, 100, True) #Bool sets whether smoothing is applied. less false positives if multimodality test is done on unsmoothed data
