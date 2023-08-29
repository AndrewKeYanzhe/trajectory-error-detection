import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.cluster.vq import kmeans

# Generate some example data
# data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(0, 1, 500)]) #1 mode
# data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500)]) #2 modes
# data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500),  np.random.normal(10, 1, 500)], ) #3 modes
data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500),  np.random.normal(10, 1, 500), np.random.normal(15, 1, 500)], ) #4 modes

def find_modes(data, show_graph, zvel_current):
    print("number of z values in filtered range:")
    print(len(data))

    multimodal = False

    # Plot the histogram
    plt.subplot(1, 2, 2)
    plt.hist(data, bins=30, density=True, alpha=0.6)

    # Compute kurtosis to measure the tailness of the distribution
    kurt = kurtosis(data)
    print("Kurtosis:", kurt)

    # Perform k-means clustering to find the number of modes
    def estimate_modes(data, k_range):
        distortions = []
        for k in k_range:
            centroids, distortion = kmeans(data.reshape(-1, 1), k)
            distortions.append(distortion)
        
        return distortions

    k_range = range(1, 11)  # Number of modes to consider



    distortions = estimate_modes(data, k_range)


    

    # first_derivative = np.gradient(distortions)


    # np.set_printoptions(suppress=True) #disable scientific format
    # # print(first_derivative)

    # indices = np.where(np.abs(first_derivative) > 0.25)[0]





    # for index in indices:
    #     print("Index:", index, "Value:", first_derivative[index])

    # number_of_modes = len(indices)

    # print("Number of modes")
    # print(number_of_modes) #this value seems to be wrong


    #-----------plotting graph with elbow

    # plt.subplot(1, 3, 3)
    # plt.plot(k_range, distortions, marker='o')
    # plt.plot(k_range, first_derivative, color = "lightblue")
    # plt.xlabel('Number of Modes')
    # plt.ylabel('Distortion')
    # plt.title('Elbow Method for Mode Estimation')
    # plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar

    # plt.show(block=False)


    kurt_thresh = 3 * max(1, min(abs(zvel_current),2))
    print("kurt threshold",kurt_thresh)



    # Decide based on kurtosis and elbow method
    if abs(kurt) > kurt_thresh and np.argmin(distortions) != 0:  #1 to minimise false positives, at the expense of slight false negatives
        print("The data appears to be bimodal or multimodal.")
        multimodal = True
    else:
        print("The data does not appear to be bimodal.")


    plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar

    if show_graph:
        plt.show(block=False)

    return multimodal

# print(find_modes(data))
