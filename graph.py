import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

# Load the CSV file into a DataFrame, skipping the first row
csv_path = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 1.csv" #ends around -1.6m
csv_path = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 2.csv" #ends around -0.8m

csv_path = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 1.csv" #-0.2 to 0.35m
csv_path = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 2.csv" #-0.5 to 0.2m

df = pd.read_csv(csv_path, skiprows=1)

#unix time is 10 digits
#for logged time values of 17 digits, truncate 7 digits to get seconds since 1970
#data seems to be 50fps, 20ms
subsampled_df = df.iloc[::10]


# # Now, if you want to work with the subsampled data in the '.timestamp' order, you can sort the DataFrame based on '.timestamp'
# subsampled_df = subsampled_df.sort_values(by='.timestamp')



# # Access columns using dictionary-like indexing
# x = df['.x']
# y = df['.y']
# z = df['.z']
# timestamps = df['.timestamp'] 

# Now you can access the subsampled data
x = subsampled_df['.x']
y = subsampled_df['.y']
z = subsampled_df['.z']
timestamps = subsampled_df['.timestamp'] # should use the second column, .timestamp

# Normalize timestamps for color gradient
norm = colors.Normalize(vmin=min(timestamps), vmax=max(timestamps))
cmap = plt.get_cmap('Blues') #later timestamps are in blue

# Create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=timestamps, cmap=cmap, norm=norm)

# Customize the colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Timestamps')

# Set axis labels
ax.set_xlabel('.x')
ax.set_ylabel('.y')
ax.set_zlabel('.z')

# plt.imshow(mask_img)
plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar

# Show the plot
plt.show()
