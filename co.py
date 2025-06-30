import pandas as pd

merged_txt_path = 'D:/Downloads/Cricket/y8.txt'  # Use your actual file path

# Read the .txt file assuming the first line is the header
df = pd.read_csv(merged_txt_path, sep='\s+', header=0)

# Strip extra spaces from column names, if any
df.columns = df.columns.str.strip()

# Print the columns to check if they are read correctly
print(df.columns)

# Print the first few rows of the DataFrame to check if data is loaded properly
print(df.head())

# Accessing the "Image_Name" column (example)
img_name = df.iloc[0]['Image_Name']
print(img_name)
