import xarray as xr
import pandas as pd
import numpy as np

LAT_CANDIDATES = ["latitude", "lat", "nav_lat", "xc"]
LON_CANDIDATES = ["longitude", "lon", "nav_lon", "yc"]

class NetCDFLoader:
    def __init__(self, file_path: str, crop_box: tuple = None):
        """
        Initialize the loader and optionally apply lat/lon cropping.

        Parameters:
            file_path: Path to the NetCDF file
            crop_box: Optional (lat_min, lat_max, lon_min, lon_max)
        """
        self.file_path = file_path
        self.ds = xr.open_dataset(file_path)
        self.crop_box = crop_box

    def get_variable_names(self):
        """
        Returns a list of all variable names.
        This includes both data variables and coordinate variables.
        """
        return list(self.ds.variables)

    def get_data_variable_names(self):
        """
        Returns a list of data variable names.
        Data variables are those that contain actual data, excluding coordinates and attributes.
        """
        return list(self.ds.data_vars)

    def get_lat_lon_names(self):
        """Detects the names of latitude and longitude variables."""
        found_lat = next((name for name in LAT_CANDIDATES if name in self.ds.variables), None)
        found_lon = next((name for name in LON_CANDIDATES if name in self.ds.variables), None)

        if found_lat is None or found_lon is None:
            raise ValueError("Could not identify latitude and longitude variables.")
        return found_lat, found_lon

    def get_variable_info(self, variable_name: str):
        """Returns metadata about a specific variable."""
        if variable_name in self.ds.data_vars:
            var = self.ds[variable_name]
            return {
                "name": variable_name,
                "dimensions": var.dims,
                "shape": var.shape,
                "dtype": str(var.dtype),
                "attrs": var.attrs
            }
        else:
            raise ValueError(f"Variable '{variable_name}' not found in dataset.")

    def get_lat_lon_values(self, crop_box: tuple = None):
        """Returns latitude and longitude coordinate arrays."""
        lat_name, lon_name = self.get_lat_lon_names()
        lat = self.ds[lat_name]
        lon = self.ds[lon_name]
        crop_box = crop_box or self.crop_box
        if crop_box:
            lat_min, lat_max, lon_min, lon_max = crop_box
            lat_slice = slice(lat_max, lat_min) if lat.values[0] > lat.values[-1] else slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max) if lon.values[0] < lon.values[-1] else slice(lon_max, lon_min)
            lat = lat.sel({lat_name: lat_slice})
            lon = lon.sel({lon_name: lon_slice})
        return lat, lon

    def get_variable(self, variable_name: str, time_index: int = None, crop_box: tuple = None):
        """
        Returns a DataArray for the given variable, possibly cropped.

        Parameters:
            variable_name: Name of the variable
            time_index: Optional index into time dimension
            crop_box: Optional (lat_min, lat_max, lon_min, lon_max)
        """
        if variable_name not in self.ds.data_vars:
            raise ValueError(f"Variable '{variable_name}' not found in dataset.")

        var = self.ds[variable_name]
        if time_index is not None and "time" in var.dims:
            var = var.isel(time=time_index)

        crop_box = crop_box or self.crop_box
        if crop_box:
            lat_min, lat_max, lon_min, lon_max = crop_box
            #sort lat/lon in case they are not in ascending order
            lat_min, lat_max = sorted([lat_min, lat_max])
            lon_min, lon_max = sorted([lon_min, lon_max])
            #get lat/lon names
            lat_name, lon_name = self.get_lat_lon_names()

            lat_vals = self.ds[lat_name].values
            lon_vals = self.ds[lon_name].values

            lat_slice = slice(lat_max, lat_min) if lat_vals[0] > lat_vals[-1] else slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max) if lon_vals[0] < lon_vals[-1] else slice(lon_max, lon_min)

            var = var.sel({lat_name: lat_slice, lon_name: lon_slice})

        return var
    

    def get_grid_data(self, variable_name: str, time_index: int = 0, crop_box: tuple = None):
        """
        Returns 2D grids of (value_grid, lat_grid, lon_grid).
        """
        da = self.get_variable(variable_name, time_index, crop_box).squeeze()
        value_grid = da.values

        lat_name, lon_name = self.get_lat_lon_names()
        lat = da[lat_name].values
        lon = da[lon_name].values

        # Ensure lat/lon increasing and align grid
        value_grid, lat, lon = self.ensure_lat_lon_order(value_grid, lat, lon)

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        return value_grid, lat_grid, lon_grid
        
    def ensure_lat_lon_order(self, value_grid, lat_vals, lon_vals):
        """
        Ensures latitude and longitude are in increasing order,
        and flips the value grid accordingly.

        Parameters:
            value_grid (2D np.ndarray): The grid of variable values.
            lat_vals (1D np.ndarray): Latitude array.
            lon_vals (1D np.ndarray): Longitude array.

        Returns:
            Tuple of (corrected_value_grid, sorted_lat, sorted_lon)
        """
        lat_sorted = lat_vals
        lon_sorted = lon_vals
        grid = value_grid

        if lat_vals[0] > lat_vals[-1]:
            lat_sorted = lat_vals[::-1]
            grid = grid[::-1, :]  # flip along latitude

        if lon_vals[0] > lon_vals[-1]:
            lon_sorted = lon_vals[::-1]
            grid = grid[:, ::-1]  # flip along longitude

        return grid, lat_sorted, lon_sorted

    def get_flat_dataframe(self, variable_name: str, time_index: int = None, crop_box: tuple = None) -> pd.DataFrame:
        """
        Returns flattened DataFrame with [lat, lon, value].
        """
        var = self.get_variable(variable_name, time_index, crop_box)
        df = var.to_dataframe(name=variable_name).reset_index()
        df = df.dropna(subset=[variable_name])
        lat_name, lon_name = self.get_lat_lon_names()
        df = df.rename(columns={lat_name: "lat", lon_name: "lon"})

        #sort dataframe by lat/lon
        df = df.sort_values(by=["lat", "lon"]).reset_index(drop=True)

        return df[["lat", "lon", variable_name]]

    def get_gridded_dataframe(self, variable_name: str, time_index: int = 0, crop_box: tuple = None) -> pd.DataFrame:
        """
        Returns a gridded DataFrame (pivot table) where:
            - Rows = latitude
            - Columns = longitude
            - Cells = variable values

        Parameters:
            variable_name: name of the variable to extract
            time_index: index into time dimension if needed
            crop_box: optional (lat_min, lat_max, lon_min, lon_max)

        Returns:
            pd.DataFrame with latitudes as index and longitudes as columns
        """
        value_grid, lat_grid, lon_grid = self.get_grid_data(variable_name, time_index, crop_box)

        # Flatten everything for pivoting
        df = pd.DataFrame({
            "lat": lat_grid.ravel(),
            "lon": lon_grid.ravel(),
            variable_name: value_grid.ravel()
        })

        # Create pivot table: rows = lat, columns = lon
        grid_df = df.pivot(index="lat", columns="lon", values=variable_name)

        return grid_df
    
    def get_numpy_inputs(self, variable_name: str, time_index: int = None, crop_box: tuple = None):
        """
        Returns (X, y) where:
            - X.shape = (N, 2) [lat, lon]
            - y.shape = (N,) variable values
        """
        df = self.get_flat_dataframe(variable_name, time_index, crop_box)
        X = df[["lat", "lon"]].values
        y = df[variable_name].values
        return X, y


    def close(self):
        """Closes the dataset."""
        self.ds.close()


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/mean_sea_level_pressure_0_daily-mean.nc"
    coordinate_box = (22, 31.7, 24.7, 36.9)  # (lat_min, lat_max, lon_min, lon_max)
    #coordinate_box = None  # No cropping by default
    loader = NetCDFLoader(data_path, crop_box=coordinate_box)#
    #variable names including coordinates
    print(loader.get_variable_names())
    #data variable names
    print(loader.get_data_variable_names())
    #coordinates names
    print(loader.get_lat_lon_names())
    
    #get lat/lon values
    lat, lon = loader.get_lat_lon_values()
    print(lat.shape, lon.shape)
    #check if np array

    #get variable data
    variable_name = loader.get_data_variable_names()[0]
    variable_data = loader.get_variable(variable_name)
    print(variable_data.shape)

    #get grid data
    value_grid, lat_grid, lon_grid = loader.get_grid_data(variable_name)
    print(value_grid.shape, lat_grid.shape, lon_grid.shape)

    #get flat dataframe
    flat_df = loader.get_flat_dataframe(variable_name)
    print(flat_df.head())

    #get gridded dataframe
    gridded_df = loader.get_gridded_dataframe(variable_name)
    print(gridded_df.head())
    

    #save dataframes to csv
    flat_df.to_csv("flat_dataframe.csv", index=False)
    gridded_df.to_csv("gridded_dataframe.csv")