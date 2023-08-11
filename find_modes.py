import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.cluster.vq import kmeans

# Generate some example data
data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(0, 1, 500)]) #1 mode
# data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500)]) #2 modes
# data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500),  np.random.normal(10, 1, 500)], ) #3 modes
# data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500),  np.random.normal(10, 1, 500), np.random.normal(15, 1, 500)], ) #4 modes


# Plot the histogram
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

first_derivative = np.gradient(distortions)
second_derivative = np.gradient(first_derivative)
third_derivative = np.gradient(second_derivative)
fourth_derivative = np.gradient(third_derivative)

np.set_printoptions(suppress=True) #disable scientific format
print(first_derivative)
print(second_derivative)
print(third_derivative)
print(fourth_derivative)

indices = np.where(np.abs(first_derivative) > 0.25)[0]

for index in indices:
    print("Index:", index, "Value:", first_derivative[index])

print("Number of modes")
print(len(indices))


plt.figure()
plt.plot(k_range, distortions, marker='o')
plt.plot(k_range, first_derivative, color = "lightblue")
plt.xlabel('Number of Modes')
plt.ylabel('Distortion')
plt.title('Elbow Method for Mode Estimation')
plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar

plt.show()

# Decide based on kurtosis and elbow method
if kurt > 0 and np.argmin(distortions) != 0:
    print("The data appears to be bimodal.")
else:
    print("The data does not appear to be bimodal.")


plt.rcParams['keymap.quit'].append(' ') #default is q. now you can close with spacebar

plt.show()
