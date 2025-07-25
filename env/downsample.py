import numpy as np
import matplotlib.pyplot as plt

def downsample(obs, factor):
    # Ensure the observation is a numpy array
    obs = np.asarray(obs)
    # Downsample the observation
    downsampled_obs = obs[::factor]
    return downsampled_obs


def main():

    # Load the observation data from a CSV file
    file_path = "env/Animal06_110919_00_31.csv"
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    # Downsample the observation by a factor of 2
    factor = 10
    downsampled_data = downsample(data, factor)

    # Save the downsampled data to a new CSV file
    downsampled_file_path = "env/Animal06_110919_00_31_downsampled.csv"
    np.savetxt(downsampled_file_path, 
               downsampled_data, 
               delimiter=',',
               fmt='%.6f',
               header=','.join(['LF_sup', 'LM_sup', 'LH_sup', 'RF_sup', 'RM_sup', 'RH_sup',
                                'LF_CTr', 'LM_CTr', 'LH_CTr', 'RF_CTr', 'RM_CTr', 'RH_CTr',
                                'LF_ThC', 'LM_ThC', 'LH_ThC', 'RF_ThC', 'RM_ThC', 'RH_ThC',
                                'LF_FTi', 'LM_FTi', 'LH_FTi', 'RF_FTi', 'RM_FTi', 'RH_FTi']), 
               comments='')
    
    # Print the original and downsampled data shapes
    print(f"Original data shape: {data.shape}")
    print(f"Downsampled data shape: {downsampled_data.shape}")

    plt.figure(figsize=(10, 5))
    plt.plot(downsampled_data[137:207,:], label='Original Data', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()