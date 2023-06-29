import pandas as pd
import os

# Set the input and output paths
INPUTS = os.path.join(os.path.dirname(os.getcwd()), "data")
OUTPUTS = os.path.join(os.path.dirname(os.getcwd()), "data")

# Load the CSV file into a pandas dataframe
filename = os.path.join(INPUTS, "validation_data.csv")
data = pd.read_csv(filename)

# Separate the data into two dataframes based on the label column
fake_val_data = data[data["overall_class"] == 0]
real_val_data = data[data["overall_class"] == 1]

# Save each dataframe to a separate CSV file
fake_val_filename = os.path.join(OUTPUTS, "fake_val.csv")
real_val_filename = os.path.join(OUTPUTS, "real_val.csv")

fake_val_data.to_csv(fake_val_filename, index=False)
real_val_data.to_csv(real_val_filename, index=False)
