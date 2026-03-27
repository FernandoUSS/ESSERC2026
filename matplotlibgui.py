import matplotlib.pyplot as plt
import numpy as np

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

plt.title("Click on the plot")

# 👇 THIS is where you click
point = plt.ginput(1)
x_click, y_click = point[0]
print(f"You clicked at: ({x_click:.2f}, {y_click:.2f})")
# Add circle
circle = plt.Circle((x_click, y_click), 0.3, color='red', fill=False)
ax.add_patch(circle)

# Add annotation
ax.annotate("Point", (x_click, y_click), xytext=(5, 5),
            textcoords='offset points')
