import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

#cal 1 is calibrated, cal 2 is naive transformation assuming zero roll and pitch offsets.
#using blue for cal 2, red for cal 1

#session 1
csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 1.csv" #ends around -1.6m
csv_path_2 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 2.csv" #ends around -0.8m. this seems to be better

#session 2
# csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 1.csv" #-0.2 to 0.35m
# csv_path_2 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 2.csv" #-0.5 to 0.2m




def read_subsampled_csv(csv_path):
    # Load the CSV file into a DataFrame, skipping the first row
    df = pd.read_csv(csv_path, skiprows=1)

    # unix time is 10 digits
    # for logged time values of 17 digits, truncate 7 digits to get seconds since 1970
    # data seems to be 50fps, 20ms
    subsampled_df = df.iloc[::10]

    # Extract data from the subsampled dataframe
    x = subsampled_df['.x']
    y = subsampled_df['.y']
    z = subsampled_df['.z']
    timestamps = subsampled_df['.timestamp']  # should use the second column, .timestamp
    
    return x, y, z, timestamps


x1, y1, z1, timestamp1 = read_subsampled_csv(csv_path_1)
x2, y2, z2, timestamp2 = read_subsampled_csv(csv_path_2)


# Normalize timestamps for color gradient
norm1 = colors.Normalize(vmin=min(timestamp1), vmax=max(timestamp1))
cmap1 = plt.get_cmap('Reds') #later timestamps are in blue

norm2 = colors.Normalize(vmin=min(timestamp2), vmax=max(timestamp2))
cmap2 = plt.get_cmap('Blues') #later timestamps are in blue

# Create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter1 = ax.scatter(x1, y1, z1, c=timestamp1, cmap=cmap1, norm=norm1)
scatter2 = ax.scatter(x2, y2, z2, c=timestamp2, cmap=cmap2, norm=norm2)


# Customize the colorbar
cbar = plt.colorbar(scatter1)
cbar.set_label('Timestamps')

# Set axis labels
ax.set_xlabel('.x')
ax.set_ylabel('.y')
ax.set_zlabel('.z')

# plt.imshow(mask_img)
plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar

# Show the plot
plt.show()
