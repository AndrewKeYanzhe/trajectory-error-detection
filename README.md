# Trajectory error detection
This script detects errors in the trajectory of a ground vehicle by detecting drift in Z height when a vehicle passes over a previous point.

The algorithm does the following:
1. Relative to the current x, y, z coordinate, filter for points in the trajectory history that is within an x, y radius
2. Calculate a histogram of the filtered points (including current point)
3. Check if the histogram has multiple, separated peaks
4. Calculate the Z height drift rate (per minute and per unit distance travelled)
