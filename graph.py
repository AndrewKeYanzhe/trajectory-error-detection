import pandas as pd
import matplotlib.pyplot as plt #backend is QtAgg on Windows 10
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

import find_modes


# This will suppress all warnings
warnings.filterwarnings("ignore")


# unix time is 10 digits
# for logged time values of 17 digits, truncate 7 digits to get seconds since 1970
# data seems to be 50fps, 20ms


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

auto_increment = False
highlight_cumulative_overlap = False

show_second_plot = True


import statsmodels.api as sm

def read_csv(csv_path, position_percent=100, smooth=False):

    # Load the CSV file into a DataFrame, skipping the first row
    df = pd.read_csv(csv_path, skiprows=1)

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


multimodal_timestamps = []

x4 = pd.Series()
y4 = pd.Series()
z4 = pd.Series()
timestamp4 = pd.Series()

history_position = 1
while True:
    print("\n")
    if auto_increment != True: 
        history_position = input("Enter position to replay until (0-100):\n")
        if history_position=="": history_position=100
        elif history_position == "a":
            history_position = 1
            auto_increment = True
            continue
    
    if auto_increment:
        history_position +=1
        print(history_position)
    

    t0 = time.time()

    if int(history_position) > 100 and auto_increment:
        break
    

    #this reads until the end position set by the user
    x1, y1, z1, xvel1, yvel1, zvel1, timestamp1 = read_csv(csv_path_1, history_position, True) #Bool sets whether smoothing is applied. less false positives if multimodality test is done on unsmoothed data
    if show_second_plot: x2, y2, z2, xvel2, yvel2, zvel1, timestamp2 = read_csv(csv_path_2, history_position, True)





    # Create a DataFrame from the lists
    data = {'x': x1, 'y': y1, 'z': z1, 'timestamp': timestamp1}
    df = pd.DataFrame(data)

    current_coordinates = (x1.iloc[-1], y1.iloc[-1], z1.iloc[-1])


    filter_size = 2

    print("x direction cm/s")
    xvel_current = xvel1.iloc[-1]*100 
    print(xvel_current)

    print("y direction cm/s")
    yvel_current = yvel1.iloc[-1]*100 
    print(yvel_current)

    print("z direction cm/s")
    zvel_current = zvel1.iloc[-1]*100 
    print(zvel_current)


    # # Filter the rows based on your criteria
    # filtered_df = df[
    #     (df['x'] >= current_coordinates[0] - filter_size_x) & 
    #     (df['x'] <= current_coordinates[0] + filter_size_x) & 
    #     (df['y'] >= current_coordinates[1] - filter_size_y) & 
    #     (df['y'] <= current_coordinates[1] + filter_size_y)
    # ]






    direction_vector = np.array([xvel1.iloc[-1], yvel1.iloc[-1]])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    print("directional_vector")
    print(direction_vector)
    print("current coordinates")
    print(current_coordinates[:2])

    perpendicular_to_travel = np.array([-direction_vector[1], direction_vector[0]])


    # Calculate perpendicular distance along the direction of travel
    df['dist_along_travel_dir'] = np.abs((df[['x', 'y']].values - current_coordinates[:2]).dot(direction_vector))
    
    df_clean = df.dropna(subset=['dist_along_travel_dir'])


    print(df_clean['dist_along_travel_dir'])

    # Calculate perpendicular distance perpendicular to the direction of travel
    df['dist_perpendicular_travel_dir'] = np.abs((df[['x', 'y']].values - current_coordinates[:2]).dot(perpendicular_to_travel))
    df_clean = df.dropna(subset=['dist_perpendicular_travel_dir'])

    print(df_clean['dist_perpendicular_travel_dir'])

    if np.linalg.norm(direction_vector) > 0.1:
        reduced_filter_size = 0.5
    else:
        reduced_filter_size = 2

    # Filter based on conditions
    filtered_df = df[(df['dist_along_travel_dir'] < reduced_filter_size) & (df['dist_perpendicular_travel_dir'] < 2)]
    print(filtered_df)






    # Extract the filtered values into new lists
    x3 = filtered_df['x']
    y3 = filtered_df['y']
    z3 = filtered_df['z']
    timestamp3 = filtered_df['timestamp']

    print("number of datapoints")
    print(len(x1))

    # subsample_factor = 20
    print("subsample factor")
    subsample_factor = max(int(len(x1)/1000),1)
    
    print(subsample_factor)

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
    fig.suptitle(history_position)

    ax = fig.add_subplot(121, projection='3d')
    scatter1 = ax.scatter(x1_sub, y1_sub, z1_sub, c=timestamp1_sub, cmap=cmap1, norm=norm1)
    if show_second_plot: scatter2 = ax.scatter(x2_sub, y2_sub, z2_sub, c=timestamp2_sub, cmap=cmap2, norm=norm2)


    multimodal = None



    xyz_coordinates = np.column_stack((x3, y3, z3))
    # xyz_coordinates = np.column_stack((y3,z3))

    # Specify the number of clusters you want
    num_clusters = 2

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(xyz_coordinates)

    # Get the labels assigned to each data point
    labels = kmeans.labels_

    # Compute the silhouette score
    silhouette_avg = silhouette_score(xyz_coordinates, labels)
    # print("Silhouette Score:\n", silhouette_avg)
    print("Silhouette Score:")
    print("{:.2f}".format(silhouette_avg))


    to_fit= np.vstack((x3,y3,z3))

    # # Calculate the minimum and maximum values
    # min_val = np.min(to_fit)
    # max_val = np.max(to_fit)

    # # Normalize to the range of -1 to 1
    # normalized_array = -1 + 2 * (to_fit - min_val) / (max_val - min_val)

    # to_fit = normalized_array


    dimensions = to_fit.shape
    print("Dimensions:", dimensions)

    to_fit=pd.DataFrame(to_fit) #converting into data frame for ease

    KMean= KMeans(n_clusters=2)
    KMean.fit(to_fit)
    label=KMean.predict(to_fit)

    silh_score = silhouette_score(to_fit, label)

    print(f'Silhouette Score(n=2): {silh_score}')

    # print(KMean.cluster_centers_)


    # Print the variance
    print("Variance in Z:", z3.var())

    

    movement_threshold = 10

    if x1.max()-x1.min() > movement_threshold or y1.max()-y1.min() > movement_threshold: 
        multimodal = find_modes.find_modes(np.array(z3), not auto_increment) #bool sets whether graph is shown
        
        pass

    if silh_score > 0.5 and z3.var()>0.0001*max(1, abs(zvel_current)):
        multimodal = True
    else:
        multimodal = False

    t1 = time.time()

    if multimodal is not None:
        x3_sub = x3.iloc[::subsample_factor]
        y3_sub = y3.iloc[::subsample_factor]
        z3_sub = z3.iloc[::subsample_factor]
        timestamp3_sub = timestamp3.iloc[::subsample_factor]


    if multimodal:
        multimodal_timestamps.append(history_position)
        if highlight_cumulative_overlap:
            x4 = x4.append(x3)
            y4 = y4.append(y3)
            z4 = z4.append(z3)
            timestamp4 = timestamp4.append(timestamp3)
        else:
            x4 = pd.concat([x4, x1[-100:]])
            y4 = pd.concat([y4, y1[-100:]])
            z4 = pd.concat([z4, z1[-100:]])
            timestamp4 = pd.concat([timestamp4, timestamp1[-100:]])
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

    

    
    
    
    if auto_increment:
        plt.close("all")
        pass
        # plt.show(block=False)
        # plt.close()
    else:
        # Show the plot
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()




    print(multimodal)
    print("Calculation time: {:.2f} s".format(t1 - t0)) #about 0.11-0.25s



if auto_increment:
    print("positions in history where multimodality is detected")
    print(multimodal_timestamps)

    fig = plt.figure()
    fig.suptitle(history_position)

    # Normalize timestamps for color gradient
    norm1 = colors.Normalize(vmin=min(timestamp1_sub), vmax=max(timestamp1_sub))
    cmap1 = plt.get_cmap('Blues') #later timestamps are in blue

    subsample_factor = 1

    x4_sub = x4.iloc[::subsample_factor]
    y4_sub = y4.iloc[::subsample_factor]
    z4_sub = z4.iloc[::subsample_factor]
    timestamp4_sub = timestamp4.iloc[::subsample_factor]


    ax = fig.add_subplot(111, projection='3d')
    scatter1 = ax.scatter(x1_sub, y1_sub, z1_sub, c=timestamp1_sub, cmap=cmap1, norm=norm1)
    scatter4 = ax.scatter(x4_sub, y4_sub, z4_sub, c="orange", zorder=99, s=100)
    plt.show()