import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('your_file.csv', header=None) # Replace 'your_file.csv' with your file path

# Assign column names for clarity (optional)
df.columns = ['Column1', 'Column2', 'Column3', 'Column4']

# Plotting
# This is a simple line plot for demonstration. 
# You might want to change the kind of plot based on your data and what you want to visualize.
plt.figure(figsize=(10, 6)) # Set the figure size
plt.plot(df['Column1'], label='Column 1') # Plotting just one column as an example
plt.title('Your Plot Title') # Add a title
plt.xlabel('X-axis Label') # Add an x-axis label
plt.ylabel('Y-axis Label') # Add a y-axis label
plt.legend() # Show legend
plt.show()
