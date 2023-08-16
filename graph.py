import pandas as pd
import matplotlib.pyplot as plt #backend is QtAgg on Windows 10
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import numpy as np
import time

import find_modes




#cal 1 is calibrated, cal 2 is naive transformation assuming zero roll and pitch offsets.
#using blue for cal 2
#using red  for cal 1

# #session 1
# csv_path_2 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 1.csv" #ends around -1.6m
# csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 2.csv" #ends around -0.8m. this seems to be better. error flagged from 29% onwards

# #session 2
csv_path_2 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 1.csv" #-0.2 to 0.35m. this seems to be an ideal results
csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 2.csv" #-0.5 to 0.2m. this has error, beginning around 45%


# csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 2.csv" #ends around -0.8m. this seems to be better

show_second_plot = True


def read_subsampled_csv(csv_path, position_percent=100):

    # Load the CSV file into a DataFrame, skipping the first row
    df = pd.read_csv(csv_path, skiprows=1)



    # unix time is 10 digits
    # for logged time values of 17 digits, truncate 7 digits to get seconds since 1970
    # data seems to be 50fps, 20ms
    
    # non_trimmed = df.iloc[::10] #subsample by a factor of 10
    index_position = int(len(df)* float(position_percent)/100)-1 
    trimmed_df = df.iloc[:index_position:]
    subsampled_df = trimmed_df.iloc[::] #subsample by a factor of 10
    

    # Extract data from the subsampled dataframe
    x = subsampled_df['.x']
    y = subsampled_df['.y']
    z = subsampled_df['.z']
    timestamps = subsampled_df['.timestamp']  # should use the second column, .timestamp
    
    return x, y, z, timestamps





while True:
    print("\n")
    user_input = input("Enter position to replay until (0-100):\n")
    if user_input=="": user_input=100
    t0 = time.time()

    

    # print(index_position)


    x1, y1, z1, timestamp1 = read_subsampled_csv(csv_path_1, user_input)
    if show_second_plot: x2, y2, z2, timestamp2 = read_subsampled_csv(csv_path_2, user_input)



    # Create a DataFrame from the lists
    data = {'x': x1, 'y': y1, 'z': z1, 'timestamp': timestamp1}
    df = pd.DataFrame(data)

    current_coordinates = (x1.iloc[-1], y1.iloc[-1], z1.iloc[-1])


    filter_size = 2

    # Filter the rows based on your criteria
    filtered_df = df[
        (df['x'] >= current_coordinates[0] - filter_size) & 
        (df['x'] <= current_coordinates[0] + filter_size) & 
        (df['y'] >= current_coordinates[1] - filter_size) & 
        (df['y'] <= current_coordinates[1] + filter_size)
    ]

    # Extract the filtered values into new lists
    x3 = filtered_df['x']
    y3 = filtered_df['y']
    z3 = filtered_df['z']
    timestamp3 = filtered_df['timestamp']

    subsample_factor = 20

    x1_sub = x1.iloc[::subsample_factor]
    y1_sub = y1.iloc[::subsample_factor]
    z1_sub = z1.iloc[::subsample_factor]
    timestamp1_sub = timestamp1.iloc[::subsample_factor]

    if show_second_plot:
        x2_sub = x2.iloc[::subsample_factor]
        y2_sub = y2.iloc[::subsample_factor]
        z2_sub = z2.iloc[::subsample_factor]
        timestamp2_sub = timestamp2.iloc[::subsample_factor]

    # print(x2_sub)


    # Normalize timestamps for color gradient
    norm1 = colors.Normalize(vmin=min(timestamp1_sub), vmax=max(timestamp1_sub))
    cmap1 = plt.get_cmap('Blues') #later timestamps are in blue

    norm2 = colors.Normalize(vmin=min(timestamp2_sub), vmax=max(timestamp2_sub)) if show_second_plot else None
    cmap2 = plt.get_cmap('Reds') #later timestamps are in blue



    # Create the 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    scatter1 = ax.scatter(x1_sub, y1_sub, z1_sub, c=timestamp1_sub, cmap=cmap1, norm=norm1)
    if show_second_plot: scatter2 = ax.scatter(x2_sub, y2_sub, z2_sub, c=timestamp2_sub, cmap=cmap2, norm=norm2)


    multimodal = None
    if len(x1)> 0: multimodal = find_modes.find_modes(np.array(z3))

    t1 = time.time()

    if multimodal is not None:
        x3_sub = x3.iloc[::subsample_factor]
        y3_sub = y3.iloc[::subsample_factor]
        z3_sub = z3.iloc[::subsample_factor]
        timestamp3_sub = timestamp3.iloc[::subsample_factor]


    if multimodal == True:
        scatter3 = ax.scatter(x3_sub, y3_sub, z3_sub, c="orange", zorder=99, s=100)
    elif multimodal == False:
        ax.scatter(x3_sub, y3_sub, z3_sub, c="green", zorder=99, s=100)



    # # Customize the colorbar  #todo can enable later
    # plt.colorbar(scatter1) if 'scatter1' in locals() else None
    # cbar.set_label('Timestamps') if 'cbar' in locals() else None

    # Set axis labels
    ax.set_xlabel('.x')
    ax.set_ylabel('.y')
    ax.set_zlabel('.z')

    # plt.imshow(mask_img)
    plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar

    # Show the plot
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()



    print(multimodal)
    print("Calculation time: {:.2f} s".format(t1 - t0)) #about 0.11-0.25s
