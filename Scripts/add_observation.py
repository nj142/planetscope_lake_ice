import netCDF4 as nc
import numpy as np

def update_lake_observation(netcdf_file, lake_id, lake_data=None, observation_data=None):
    """
    Update a NetCDF file with a new observation for a specific lake.
    If the lake doesn't exist, it creates a new lake entry.
    If the observation with the exact datetime already exists, it skips the update.
    
    Parameters:
    -----------
    netcdf_file : str
        Path to the NetCDF file
    lake_id : int
        ID of the lake to update
    lake_data : dict or None
        Dictionary containing lake-level data if this is a new lake:
        - area: float
        - perimeter: float
        - study_site: str
        - total_pixels: int
        If None and lake doesn't exist, raises an error
    observation_data : dict
        Dictionary containing observation data:
        - datetime: int (Unix timestamp)
        - prefix: str
        - usable_pixels: int
        - clear_percent: float
        - ice_pixels: int
        - ice_percent: float
        - snow_pixels: int
        - snow_percent: float
        - water_pixels: int
        - water_percent: float
    
    Returns:
    --------
    str : Status message indicating what action was taken
    """
    with nc.Dataset(netcdf_file, 'r+') as ds:
        # Get current dimensions
        lake_dim = ds.dimensions['lake'].size
        obs_dim = ds.dimensions['obs'].size
        
        # Check if lake exists
        lake_ids = ds.variables['lake_id'][:]
        lake_idx = np.where(lake_ids == lake_id)[0]
        
        lake_exists = len(lake_idx) > 0
        
        # If lake doesn't exist, create it
        if not lake_exists:
            if lake_data is None:
                raise ValueError(f"Lake ID {lake_id} not found and no lake_data provided to create it")
            
            # Verify lake_data has all required fields
            required_fields = ['area', 'perimeter', 'study_site', 'total_pixels']
            for field in required_fields:
                if field not in lake_data:
                    raise ValueError(f"Missing required field '{field}' in lake_data")
            
            # Add the new lake data to each lake-level variable
            lake_idx = lake_dim  # Index of the new lake
            
            # Add the new lake to lake variables
            for var_name in ['lake_id', 'area', 'perimeter', 'study_site', 'total_pixels']:
                var = ds.variables[var_name]
                
                if var.dtype == str:
                    # For string variables
                    var[lake_dim] = lake_data[var_name] if var_name != 'lake_id' else str(lake_id)
                else:
                    # For numeric variables
                    current_data = var[:]
                    value = lake_data[var_name] if var_name != 'lake_id' else lake_id
                    new_data = np.append(current_data, value)
                    var[:] = new_data
            
            # Initialize count to 0 for the new lake
            count_var = ds.variables['count']
            current_counts = count_var[:]
            new_counts = np.append(current_counts, 0)
            count_var[:] = new_counts
            
            status = f"Created new lake ID {lake_id}"
        else:
            lake_idx = lake_idx[0]  # Get the actual index
            status = f"Found existing lake ID {lake_id} at index {lake_idx}"
        
        # Check if this exact observation already exists
        if lake_exists:
            # Get the range of observations for this lake
            counts = ds.variables['count'][:]
            lake_obs_start = 0 if lake_idx == 0 else np.sum(counts[:lake_idx])
            lake_obs_end = lake_obs_start + counts[lake_idx]
            
            # Get the datetimes for this lake's observations
            lake_datetimes = ds.variables['datetime'][lake_obs_start:lake_obs_end]
            
            # Check if the observation datetime already exists
            if observation_data['datetime'] in lake_datetimes:
                return status + f", observation with datetime {observation_data['datetime']} already exists, skipping"
        
        # Add the new observation
        # Calculate insert position
        if lake_exists:
            counts = ds.variables['count'][:]
            if lake_idx == 0:
                insert_pos = 0
            else:
                insert_pos = np.sum(counts[:lake_idx])
            
            # Add the insert position to the count for this lake
            insert_pos += counts[lake_idx]
        else:
            # For a new lake, just add to the end
            insert_pos = obs_dim
        
        # Extend all observation variables by 1
        obs_variables = ['lake_index', 'datetime', 'prefix', 'usable_pixels', 
                        'clear_percent', 'ice_pixels', 'ice_percent', 
                        'snow_pixels', 'snow_percent', 'water_pixels', 
                        'water_percent']
        
        for var_name in obs_variables:
            var = ds.variables[var_name]
            
            if var_name == 'lake_index':
                value = lake_idx
            else:
                value = observation_data[var_name]
            
            if var.dtype == str:
                # For string variables, we need to shift all values after insert_pos
                # First, get the current data
                if lake_exists:
                    # Shift all values after insert_pos
                    for i in range(obs_dim - 1, insert_pos - 1, -1):
                        var[i + 1] = var[i]
                    # Then insert the new value
                    var[insert_pos] = value
                else:
                    # For a new lake, just append
                    var[obs_dim] = value
            else:
                # For numeric variables
                current_data = var[:]
                if lake_exists:
                    new_data = np.insert(current_data, insert_pos, value)
                else:
                    new_data = np.append(current_data, value)
                var[:] = new_data
        
        # Update the count for this lake
        ds.variables['count'][lake_idx] += 1
        
        return status + f", added new observation with datetime {observation_data['datetime']}"

# Example usage
if __name__ == "__main__":
    # Path to your NetCDF file
    netcdf_file = r"D:\planetscope_lake_ice\Data (Unclassified)\2 - Break Up Time Series Output\lake_statistics_new.nc"
    
    # Example for an existing lake (modify as needed)
    lake_id = 183702
    observation_data = {
        'datetime': 1622760000,  # New unique timestamp (June 3, 2021)
        'prefix': '20210603_123456_123_3B_AnalyticMS_SR',
        'usable_pixels': 206,
        'clear_percent': 100.0,
        'ice_pixels': 0,
        'ice_percent': 0.0,
        'snow_pixels': 0,
        'snow_percent': 0.0,
        'water_pixels': 206,
        'water_percent': 100.0
    }
    
    # Optional: data for creating a new lake if needed
    lake_data = {
        'area': 1895.5,
        'perimeter': 224.76,
        'study_site': 'YKD',
        'total_pixels': 206
    }
    
    result = update_lake_observation(netcdf_file, lake_id, lake_data, observation_data)
    print(result)
    
    # Example for a new lake (uncomment to test)
    """
    new_lake_id = 999999
    new_lake_data = {
        'area': 2500.0,
        'perimeter': 200.0,
        'study_site': 'YKD',
        'total_pixels': 240
    }
    
    new_observation = {
        'datetime': 1622500000,
        'prefix': '20210601_123456_123_3B_AnalyticMS_SR',
        'usable_pixels': 240,
        'clear_percent': 100.0,
        'ice_pixels': 200,
        'ice_percent': 83.33,
        'snow_pixels': 40,
        'snow_percent': 16.67,
        'water_pixels': 0,
        'water_percent': 0.0
    }
    
    result = update_lake_observation(netcdf_file, new_lake_id, new_lake_data, new_observation)
    print(result)
    """