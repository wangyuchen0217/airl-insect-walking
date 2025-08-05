import pandas as pd

# Concatenate the csv files expert0~expert29.csv
for i in range(30):
    expert_file = f'expert/trails/expert{i}.csv'
    data = pd.read_csv(expert_file, header=[0])
    if i == 0:
        all_data = data
    else:
        all_data = pd.concat([all_data, data], ignore_index=True)
print(f"Concatenated data shape: {all_data.shape}")

# Save the concatenated data to a new CSV file
all_data.to_csv('expert/expert_60000.csv', index=False)