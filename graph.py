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
from scipy.signal import find_peaks

import find_modes


# This will suppress all warnings
warnings.filterwarnings("ignore")

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def generate_histogram(data, num_bins=10):
    # Step 1: Find the minimum and maximum values
    if len(data)==0:
        return [], []

    min_value = min(data)
    max_value = max(data)
    
    # Step 2: Calculate the bin width
    bin_width = (max_value - min_value) / num_bins
    
    # Initialize bins and counts
    bins = [min_value + i * bin_width for i in range(num_bins + 1)]
    bin_counts = [0] * num_bins
    
    # Step 3: Count occurrences in bins
    for value in data:
        for i in range(num_bins):
            if bins[i] <= value < bins[i+1]:
                bin_counts[i] += 1
                break
    
    return bins, bin_counts


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
initial_history_position = 1
auto_increment_stop = 100
highlight_cumulative_overlap = False

#fps at which you check for multimodality. default is 1/3. 10fps is slow. 2fps is ok
fps = 1/3




show_second_plot = True

# Load the CSV file into a DataFrame, skipping the first row
csv1 = pd.read_csv(csv_path_1, skiprows=1)
csv2 = pd.read_csv(csv_path_2, skiprows=1)

plot_drift_vs_time = 1
#if zero, means plot drift vs dist


def read_csv(csv, position_percent=100, smooth=False):

    index_position = int(len(csv)* float(position_percent)/100)-1 
    trimmed_csv = csv.iloc[:index_position:]
    
    # Extract data from the subsampled dataframe
    x = trimmed_csv['.x']
    y = trimmed_csv['.y']
    z = trimmed_csv['.z']
    xvel = trimmed_csv['.xVelocity']
    yvel = trimmed_csv['.yVelocity']
    zvel = trimmed_csv['.zVelocity']
    timestamp = trimmed_csv['.timestamp']  # should use the second column, .timestamp

    if smooth:
        # Apply Lowess smoothing to each dimension
        smoothing_frac = 90/len(x)  # Smoothing fraction, you can adjust this value

        smoothed_x = sm.nonparametric.lowess(x, timestamp, frac=smoothing_frac, it=0)[:, 1]
        smoothed_y = sm.nonparametric.lowess(y, timestamp, frac=smoothing_frac, it=0)[:, 1]
        smoothed_z = sm.nonparametric.lowess(z, timestamp, frac=smoothing_frac, it=0)[:, 1]

        # Create Pandas Series with smoothed data and timestamps
        smoothed_x_series = pd.Series(smoothed_x, index=timestamp)
        smoothed_y_series = pd.Series(smoothed_y, index=timestamp)
        smoothed_z_series = pd.Series(smoothed_z, index=timestamp)
        return smoothed_x_series, smoothed_y_series, smoothed_z_series, xvel, yvel, zvel, timestamp
    else:
        return x, y, z, xvel, yvel, zvel, timestamp


multimodal_timestamps_silh = []
multimodal_timestamps_kurt=[]
drift_vs_dist_list_silh =[]
drift_vs_time_list_silh=[]
drift_vs_dist_list_kurt =[]
drift_vs_time_list_kurt=[]

x4_silh = pd.Series()
y4_silh = pd.Series()
z4_silh = pd.Series()
timestamp4_silh = pd.Series()


x4_kurt = pd.Series()
y4_kurt = pd.Series()
z4_kurt = pd.Series()
timestamp4_kurt = pd.Series()


t0=time.time()

history_position=1

while True:
    enablePrint()
    
    if auto_increment != True: 
        print("\n")
        history_position = input("Enter position to replay until (0-100). Enter 'a' to play through entire session\n")
        if history_position=="": history_position=100
        elif history_position == "a":
            history_position = initial_history_position
            auto_increment = True
            continue
    
    if auto_increment:
        # history_position +=1
        history_position += 50/fps/csv1.shape[0]*100 #in percent
        print("Calculation time: {:.2f} s".format(time.time() - t0)) #about 0.11-0.25s
        print("\n")
        print("history position: " + "{:.1f}".format(history_position))

        blockPrint()
    

    t0 = time.time()

    if float(history_position) > auto_increment_stop and auto_increment:
        break
    

    #this reads until the end position set by the user
    x1, y1, z1, xvel1, yvel1, zvel1, timestamp1 = read_csv(csv1, history_position, True) #Bool sets whether smoothing is applied. less false positives if multimodality test is done on unsmoothed data
    if show_second_plot: x2, y2, z2, xvel2, yvel2, zvel1, timestamp2 = read_csv(csv2, history_position, True)





    # Create a DataFrame from the lists
    data = {'x': x1, 'y': y1, 'z': z1}
    df = pd.DataFrame(data)

    # print(data)

    differences = df.diff().fillna(0)
    dist_intervals = np.sqrt(differences['x']**2 + differences['y']**2 + differences['z']**2)
    dist_travelled = sum(dist_intervals)
    print("total distance travelled:",dist_travelled)

    df['dist_intervals'] = dist_intervals

    # Print the updated DataFrame
    # print(df)

    current_coordinates = (x1.iloc[-1], y1.iloc[-1], z1.iloc[-1])


    filter_size = 2
    print("averaged over 0.5s:")
    print("x direction cm/s: ", end="")
    xvel_current = xvel1.iloc[-25:].mean()*100 
    print(xvel_current)

    print("y direction cm/s: ", end="")
    yvel_current = yvel1.iloc[-25:].mean()*100 
    print(yvel_current)

    print("z direction cm/s: ", end="")
    zvel_current = zvel1.iloc[-25:].mean()*100 #in cm per second
    print(zvel_current)


 




    direction_vector = np.array([xvel1.iloc[-25:].mean(), yvel1.iloc[-25:].mean()])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # print("directional_vector")
    # print(direction_vector)
    # print("current coordinates")
    # print(current_coordinates[:2])

    perpendicular_to_travel = np.array([-direction_vector[1], direction_vector[0]])


    # Calculate perpendicular distance along the direction of travel
    df['dist_along_travel_dir'] = (df[['x', 'y']].values - current_coordinates[:2]).dot(direction_vector)
    
    df_clean = df.dropna(subset=['dist_along_travel_dir'])
    print(df)

    print("df including dist along travel direction")
    # print(df_clean['dist_along_travel_dir'])
    print(df_clean)

    # Calculate perpendicular distance perpendicular to the direction of travel
    df['dist_perpendicular_travel_dir'] = np.abs((df[['x', 'y']].values - current_coordinates[:2]).dot(perpendicular_to_travel))
    df_clean = df.dropna(subset=['dist_perpendicular_travel_dir'])

    # print(df_clean['dist_perpendicular_travel_dir'])

    if np.linalg.norm(direction_vector) > 0.1:
        reduced_filter_size = 0.25 
    else:
        reduced_filter_size = 2

    # Filter based on conditions
    filtered_df = df[(-1*reduced_filter_size<df['dist_along_travel_dir'] ) &(df['dist_along_travel_dir'] < reduced_filter_size) & (df['dist_perpendicular_travel_dir'] < 2)]
    
    
    print("after filtering for x y position ")
    print(filtered_df)

    print #todo print max z in last second





    # Extract the filtered values into new lists
    x3 = filtered_df['x']
    y3 = filtered_df['y']
    z3 = filtered_df['z']
    # timestamp3 = filtered_df['timestamp']

    # print(filtered_df)


    bins, bin_counts = generate_histogram(z3, num_bins=30)
    

    bin_counts = [0] + bin_counts + [0]

    print("bins")
    print(bins)

    print("bin counts")
    print(bin_counts)

    zvel_1sec = zvel1.iloc[-50:].mean()*100 #cm per second
    print("avg z speed in the last minute (50 readings)",zvel_1sec)


    default_prominance_thresh = 4

    # if 0 not in bins [13:17]: default_prominance_thresh = 20

    contains_zero = 0.0 in bin_counts[13:17]

    print("contains zero:",contains_zero)

    print(type(bins[15]))

    if not contains_zero and abs(zvel_1sec)>1:
        default_prominance_thresh = 40
        #20 works decently

    if len(bins)>=2:
        peaks_dist_thresh = max(int( 0.01/(bins[-1]-bins[0])*30),1) #0.01 is 1cm

        #adjust for z velocity
        # peaks_dist_thresh = peaks_dist_thresh*max(1, min(abs(zvel_current),6))
        peaks_dist_thresh = peaks_dist_thresh
        print("distance threshold", peaks_dist_thresh)

        # prominence_thresh = default_prominance_thresh*1*max(1, min(abs(zvel_1sec),1.5))
        prominence_thresh = default_prominance_thresh
        print("prominence threshold",prominence_thresh)
    else:
        peaks_dist_thresh=None
        prominence_thresh = default_prominance_thresh



    #finding peaks
    # peaks = find_peaks(bin_counts, height=max(bin_counts)/3, prominence=2)
    peaks = find_peaks(bin_counts, prominence=prominence_thresh, distance = peaks_dist_thresh, width=1)
    
    # height = peaks[1]['peak_heights'] #list of the heights of the peaks
    # peak_pos = peaks[0]] #list of the peaks positions


    print("peak pos",peaks)
    print("multiple peaks", len(peaks[0])>=2)

    
    multiple_peaks = len(peaks[0])>=2

    #filter for points at least 5 seconds ago
    second_filtered_df = filtered_df[filtered_df.index < timestamp1.iloc[-1] - 5e7] #check for z error vs latest point, at least 5 seconds ago
    
    # print(df.iloc[-1])

    # print(second_filtered_df)

    #calculate time elapsed
    if not second_filtered_df.empty:
        time_elapsed = (timestamp1.iloc[-1] - second_filtered_df.index[-1])/1e7
        print("time elapsed:",time_elapsed)
    





    print("number of datapoints in x1")
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
    

    ax = fig.add_subplot(121, projection='3d')
    scatter1 = ax.scatter(x1_sub, y1_sub, z1_sub, c=timestamp1_sub, cmap=cmap1, norm=norm1)
    if show_second_plot: scatter2 = ax.scatter(x2_sub, y2_sub, z2_sub, c=timestamp2_sub, cmap=cmap2, norm=norm2)


    multimodal = False



    xyz_coordinates = np.column_stack((x3, y3, z3))
    if xyz_coordinates.shape == (0, 3) and auto_increment:
        continue
    

    # Specify the number of clusters you want
    num_clusters = 2

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(xyz_coordinates)

    # Get the labels assigned to each data point
    labels = kmeans.labels_

    # Compute the silhouette score
    silhouette_avg = silhouette_score(xyz_coordinates, labels) #silhouette_avg is not that good, use other silhouette score in later section
    # print("Silhouette Score:\n", silhouette_avg)
    print("Silhouette Score avg:")
    print("{:.2f}".format(silhouette_avg))


    to_fit= np.vstack((x3,y3,z3))

    # # Calculate the minimum and maximum values
    # min_val = np.min(to_fit)
    # max_val = np.max(to_fit)

    # # Normalize to the range of -1 to 1
    # normalized_array = -1 + 2 * (to_fit - min_val) / (max_val - min_val)

    # to_fit = normalized_array


    dimensions = to_fit.shape
    print("Dimensions of x3 y3 z3 (distance filtered). This is the number used for calculating silhouette clusters:", dimensions)

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

    
        
        
    #decide if data is multimodal    

    # if silh_score > 0.5 and z3.var()>0.0001*max(1, abs(zvel_current)):
    if x1.max()-x1.min() > movement_threshold or y1.max()-y1.min() > movement_threshold : 
        if multiple_peaks:
            multimodal = True #silhouette alone might miss points where the pose is drifting but robot is stationary (92 to 100). hence use both silhoutte and find_modes (strict threshold at 3)
            multimodal_timestamps_silh.append(history_position)
            if highlight_cumulative_overlap: #highlights all points that contain double Z height
                x4_silh = x4_silh.append(x3)
                y4_silh = y4_silh.append(y3)
                z4_silh = z4_silh.append(z3)
                timestamp4_silh = timestamp4_silh.append(timestamp3)
            else:
                points_to_highlight = 1 #highlights points where multimodality is detectable (so second pass or above)
                x4_silh = pd.concat([x4_silh, x1[-points_to_highlight:]])
                y4_silh = pd.concat([y4_silh, y1[-points_to_highlight:]])
                z4_silh = pd.concat([z4_silh, z1[-points_to_highlight:]])
                timestamp4_silh = pd.concat([timestamp4_silh, timestamp1[-100:]])
            
    multimodal_kurt=False

    #if silhouette didnt detect multimodality, double check using kurtosis
    if x1.max()-x1.min() > movement_threshold or y1.max()-y1.min() > movement_threshold : 
        multimodal_kurt = find_modes.find_modes(np.array(z3), not auto_increment, zvel_current) #bool sets whether graph is shown
    

    detected_by_kurt = False

    if multimodal_kurt and multimodal!=True:
        # multimodal = True
        detected_by_kurt = True
        # multimodal_timestamps_kurt.append(history_position)
        # if highlight_cumulative_overlap: #highlights all points that contain double Z height
        #     x4_kurt = x4_kurt.append(x3)
        #     y4_kurt = y4_kurt.append(y3)
        #     z4_kurt = z4_kurt.append(z3)
        #     timestamp4_kurt = timestamp4_kurt.append(timestamp3)
        # else:
        #     points_to_highlight = 1 #highlights points where multimodality is detectable (so second pass or above)
        #     x4_kurt = pd.concat([x4_kurt, x1[-points_to_highlight:]])
        #     y4_kurt = pd.concat([y4_kurt, y1[-points_to_highlight:]])
        #     z4_kurt = pd.concat([z4_kurt, z1[-points_to_highlight:]])
        #     timestamp4_kurt = pd.concat([timestamp4_kurt, timestamp1[-100:]])

    #calculate change in z
    current_z = z1.iloc[-1]
    print("current z:",current_z)
    if not second_filtered_df.empty and multimodal:
        last_z = second_filtered_df['z'].iloc[-1]
        print("last Z:", last_z)

        #calculate change in distance travelled
        # current_dist = dist_travelled
        last_index = second_filtered_df.index[-1]
        change_in_dist = df.loc[last_index:, 'dist_intervals'].sum()
        drift_vs_dist = (current_z - last_z) /change_in_dist*100 #in percent

    else:
        drift_vs_dist=0


    






    #calclate drift per minute
    if second_filtered_df.empty or not multimodal:
        drift_vs_time = 0
    else:
        drift_vs_time = (current_z - last_z )/time_elapsed*60 #per minute
    print("drift per minute:", drift_vs_time)


    if multimodal and not detected_by_kurt:
        print("detected by silhouette")
        drift_vs_dist_list_silh.append(drift_vs_dist)
        drift_vs_time_list_silh.append(drift_vs_time)

    elif multimodal and detected_by_kurt:
        print("detected by kurtosis")
        drift_vs_dist_list_kurt.append(drift_vs_dist)
        drift_vs_time_list_kurt.append(drift_vs_time)


    t1 = time.time()

    fig.suptitle("history position: " + str(history_position) + ", distance: {:.1f}".format(dist_travelled) + ", drift per minute: {:.2f}".format(drift_vs_time)+ ", change in Z for change in dist: {:.2f}".format(drift_vs_dist)+"%")

    
    # if multimodal is not None: #for graphing
    x3_sub = x3.iloc[::subsample_factor]
    y3_sub = y3.iloc[::subsample_factor]
    z3_sub = z3.iloc[::subsample_factor]
    # timestamp3_sub = timestamp3.iloc[::subsample_factor]

        
        
    


    if multimodal:
        scatter3 = ax.scatter(x3_sub, y3_sub, z3_sub, c="orange", label="overlap", zorder=99, s=100)
    elif multimodal == False :
        ax.scatter(x3_sub, y3_sub, z3_sub, c="green", label = "no overlap", zorder=99, s=100)

    ax.legend()



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
        plt.tight_layout()

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()




    print("overlap: ",multimodal)
    print("Calculation time: {:.2f} s".format(t1 - t0)) #about 0.11-0.25s

enablePrint()

if auto_increment:
    

    print("\npositions in history where multimodality is detected")
    print("detected by silhouette")
    for timestamp in multimodal_timestamps_silh:
        timestamp = float(timestamp)
        print("{:.2f}".format(timestamp))

    print("\n")

    print("additional points detected by kurtosis")
    for timestamp in multimodal_timestamps_kurt:
        timestamp = float(timestamp)
        print("{:.2f}".format(timestamp))


    # # Printing drift per unit distance with 1 decimal place
    # print("\ndrift per unit dist in %")
    # for value in drift_vs_dist_list_silh + drift_vs_dist_list_kurt:
    #     print("{:.1f}%".format(value))

    # print("\n")

    # # Printing drift per minute with 2 decimal places
    # print("drift per minute")
    # for value in drift_vs_time_list_silh + drift_vs_time_list_kurt:
    #     print("{:.2f}".format(value))

    fig = plt.figure()
    

    # Normalize timestamps for color gradient
    norm1 = colors.Normalize(vmin=min(timestamp1_sub), vmax=max(timestamp1_sub))
    cmap1 = plt.get_cmap('Blues') #later timestamps are in blue

    # subsample_factor = 1

    # x4_silh_sub = x4_silh.iloc[::subsample_factor]
    # y4_silh_sub = y4_silh.iloc[::subsample_factor]
    # z4_silh_sub = z4_silh.iloc[::subsample_factor]
    # timestamp4_sub = timestamp4.iloc[::subsample_factor]


    ax = fig.add_subplot(111, projection='3d')
    scatter1 = ax.scatter(x1_sub, y1_sub, z1_sub, c=timestamp1_sub, cmap=cmap1, norm=norm1)
    scatter5 = ax.scatter(x4_silh, y4_silh, z4_silh, c="orange", label="silhouette", zorder=99, s=100)
    scatter4 = ax.scatter(x4_kurt, y4_kurt, z4_kurt, c="brown", label="kurtosis", zorder=99, s=100)
    
    



    # print(drift_vs_time_list_silh)
    # print(x4_silh) #pandas series

    
    # note: annotate is only for 2d plot. 
    # ax.text can be used for 3d plot


    if plot_drift_vs_time:

        #plot drift rate (time) for each point where multimodality is detected
        for index, (i, j, k) in enumerate(zip(x4_silh.tolist(), y4_silh.tolist(), z4_silh.tolist())):
            label = f"{drift_vs_time_list_silh[index]:.2f}"
            ax.text(i+0.6, j+0.6, k+0.01, label)
        for index, (i, j, k) in enumerate(zip(x4_kurt.tolist(), y4_kurt.tolist(), z4_kurt.tolist())):
            label = f"{drift_vs_time_list_kurt[index]:.2f}"
            ax.text(i+0.6, j+0.6, k+0.01, label)

        fig.suptitle(f"History position: {history_position:.1f}, drift is in m/min")

    else:
        #plot drift rate (dist) for each point where multimodality is detected
        for index, (i, j, k) in enumerate(zip(x4_silh.tolist(), y4_silh.tolist(), z4_silh.tolist())):
            label = f"{drift_vs_dist_list_silh[index]:.2f}"
            ax.text(i+0.6, j+0.6, k+0.01, label)
        for index, (i, j, k) in enumerate(zip(x4_kurt.tolist(), y4_kurt.tolist(), z4_kurt.tolist())):
            label = f"{drift_vs_dist_list_kurt[index]:.2f}"
            ax.text(i+0.6, j+0.6, k+0.01, label)

        fig.suptitle(f"History position: {history_position:.1f}, labels are change in drift (z) /change in meters travelled (x,y,z), in %")
    

    ax.legend()
    plt.tight_layout()

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()