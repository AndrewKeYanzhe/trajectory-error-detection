import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

# Load the CSV file into a DataFrame, skipping the first row
csv_path = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 1.csv"
df = pd.read_csv(csv_path, skiprows=1)

# Access columns using dictionary-like indexing
x = df['.x']
y = df['.y']
z = df['.z']
timestamps = df['timestamp']  # Assuming the timestamp column is named 'timestamp'

# Normalize timestamps for color gradient
norm = colors.Normalize(vmin=min(timestamps), vmax=max(timestamps))
cmap = plt.get_cmap('Blues')

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

# Show the plot
plt.show()
