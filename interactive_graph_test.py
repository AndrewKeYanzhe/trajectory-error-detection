# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Enable interactive mode
plt.ion()

# Create a figure and axis
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a line plot
line, = ax.plot(x, y)

# Update the plot in a loop
for phase in np.linspace(0, 10, 100):
    y_new = np.sin(x + phase)
    
    # Update the y-data of the line plot
    line.set_ydata(y_new)
    
    # Redraw the figure
    fig.canvas.draw()
    
    # Pause for a short time to observe the animation
    plt.pause(0.1)

# Turn off interactive mode
plt.ioff()

# Display the final plot
plt.show()
