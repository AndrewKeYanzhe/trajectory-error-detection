import pandas as pd
import matplotlib.pyplot as plt #backend is QtAgg on Windows 10
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import numpy as np

import find_modes




#cal 1 is calibrated, cal 2 is naive transformation assuming zero roll and pitch offsets.
#using blue for cal 2
#using red  for cal 1

# #session 1
# csv_path_2 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 1.csv" #ends around -1.6m
# csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 2.csv" #ends around -0.8m. this seems to be better

#session 2
csv_path_2 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 1.csv" #-0.2 to 0.35m. this seems to be an ideal results
csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH52 TVE Sensor Log with cal 2.csv" #-0.5 to 0.2m. this has error, beginning around 45%


# csv_path_1 = r"C:\Users\kyanzhe\Downloads\lidar-imu-calibration\(2023-07-25) FH51 TVE Sensor Log with cal 2.csv" #ends around -0.8m. this seems to be better




def read_subsampled_csv(csv_path, position_percent=100):

    # Load the CSV file into a DataFrame, skipping the first row
    df = pd.read_csv(csv_path, skiprows=1)

    if 'x0' in globals(): 
        index_position = int(len(x0) * float(position_percent)/100)-1 
        subsampled_df = df.iloc[:index_position:10] #subsample by a factor of 10
    else:
        # unix time is 10 digits
        # for logged time values of 17 digits, truncate 7 digits to get seconds since 1970
        # data seems to be 50fps, 20ms
        subsampled_df = df.iloc[::10] #subsample by a factor of 10
        

    

    

    # Extract data from the subsampled dataframe
    x = subsampled_df['.x']
    y = subsampled_df['.y']
    z = subsampled_df['.z']
    timestamps = subsampled_df['.timestamp']  # should use the second column, .timestamp
    
    return x, y, z, timestamps


show_second_plot = False



x0, y0, z0, timestamp0 = read_subsampled_csv(csv_path_1)


while True:
    print("\n")
    user_input = input("Enter position in percent:\n")

    index_position = int(len(x0) * float(user_input)/100)-1

    # print(index_position)


    x1, y1, z1, timestamp1 = read_subsampled_csv(csv_path_1, index_position)
    if show_second_plot: x2, y2, z2, timestamp2 = read_subsampled_csv(csv_path_2)



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
    x3 = filtered_df['x'].tolist()
    y3 = filtered_df['y'].tolist()
    z3 = filtered_df['z'].tolist()
    timestamp3 = filtered_df['timestamp'].tolist()








    # Normalize timestamps for color gradient
    norm1 = colors.Normalize(vmin=min(timestamp1), vmax=max(timestamp1))
    cmap1 = plt.get_cmap('Blues') #later timestamps are in blue

    norm2 = colors.Normalize(vmin=min(timestamp2), vmax=max(timestamp2)) if show_second_plot else None
    cmap2 = plt.get_cmap('Reds') #later timestamps are in blue



    # Create the 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    scatter1 = ax.scatter(x1, y1, z1, c=timestamp1, cmap=cmap1, norm=norm1)
    if show_second_plot: scatter2 = ax.scatter(x2, y2, z2, c=timestamp2, cmap=cmap2, norm=norm2)

    multimodal = find_modes.find_modes(np.array(z3))


    scatter3 = ax.scatter(x3, y3, z3, c="orange",zorder=99, s=100) if multimodal else ax.scatter(x3, y3, z3, c="green",zorder=99, s=100)


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