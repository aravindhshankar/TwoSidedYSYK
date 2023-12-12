import h5py

##### Benchmarked by test_compression.py


def dict2h5(my_dict,file_path, verbose = False):
	'''
	arguments 1. dictionary 2. path to file 3. verbose?
	author : chat GPT
	'''
	# Open the HDF5 file in write mode
	with h5py.File(file_path, 'w') as hf:
	    # Create a group in the HDF5 file
	    group = hf.create_group('my_group')

	    # Iterate through the dictionary items and save them to the group
	    for key, value in my_dict.items():
	        if isinstance(value, dict):
	            # If the value is a nested dictionary, create a subgroup
	            subgroup = group.create_group(key)
	            # Iterate through the nested dictionary and save its items
	            for subkey, subvalue in value.items():
	                subgroup[subkey] = subvalue
	        else:
	            # Save non-dictionary values directly to the group
	            group[key] = value

	if(verbose):
		print(f'Dictionary saved to {file_path}')

	return


def h52dict(file_path, verbose = False):
	'''
	argument: path to file
	returns: dictionary
	'''
	# Create an empty dictionary to store the data
	loaded_dict = {}

	# Open the HDF5 file in read mode
	with h5py.File(file_path, 'r') as hf:
	    # Access the group created in the previous example
	    group = hf['my_group']

	    # Iterate through the items in the group
	    for key, value in group.items():
	        if isinstance(value, h5py.Group):
	            # If the item is a group, it means it was a nested dictionary
	            nested_dict = {}
	            # Iterate through the items in the subgroup
	            for subkey, subvalue in value.items():
	                nested_dict[subkey] = subvalue[()]
	            # Add the nested dictionary to the loaded dictionary
	            loaded_dict[key] = nested_dict
	        else:
	            # If the item is not a group, it means it was a direct value
	            loaded_dict[key] = value[()]

	if verbose:
		print('Dictionary loaded from HDF5 file:')

	return (loaded_dict)

