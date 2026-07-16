import numpy as np
import sys

# Load the file using a context manager
with np.load(sys.argv[1]) as data:
    # 1. List all array names (keys) stored inside
    print("Keys:", data.files)
    
    # 2. Inspect metadata (shapes and dtypes) without printing full data
    for key in data.files:
        print(f"Key: {key} | Shape: {data[key].shape} | Dtype: {data[key].dtype}")
        if key == 'pano':
            print(f"First 5 elements of '{key}': {data[key].flatten()[:5]}")  # Print first 5 elements of 'pano'
        
    # 3. Access a specific array
    # my_array = data['arr_0']
