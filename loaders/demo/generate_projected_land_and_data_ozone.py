import numpy as np
from utils.coordinate_transformer import CoordinateTransformer
from loaders.netcdf_loader import NetCDFLoader
import pandas as pd
import matplotlib.pyplot as plt

def generate_projected_land_and_data():
    """
    This is an example function that generates projected land polygons from cartopy data and generates a NetCDF file with data in
    the same projection. It uses the CoordinateTransformer to transform the land polygons and data coordinates.
    the coordinates used here are for the Canadian Arctic Archipelago, but you can change them to any other region.
    """

    #coordinates min/max latitude and longitude
    lat_min = 65
    lat_max = 120
    lon_min = 220
    lon_max = 330


    lat_min = 68
    lat_max = 84
    lon_min = CoordinateTransformer.wrap_lon_to_360(-110)
    lon_max = CoordinateTransformer.wrap_lon_to_360(-60)
    # Create a CoordinateTransformer instance
    transformer = CoordinateTransformer()
    # Get the transformed land polygons
    #extent = (lat_min, lat_max, lon_min, lon_max) is the bounding box for the land polygons
    #resolution="110m" is the resolution of the land polygons (using the lowest resolution for faster processing)
    #wrap_longitudes=True is used to wrap the longitudes> 180 degrees to the range [-180, 180] degrees
    land_polygons = transformer.get_transformed_land_polygons(
        extent=(lat_min, lat_max, lon_min, lon_max),
        resolution="110m",
        wrap_longitudes=True
    )

    # Convert to DataFrame
    df = pd.DataFrame({
        "x": [x for x, _ in land_polygons],
        "y": [y for _, y in land_polygons]
    })

    # save the land polygons to pkl file
    df.to_pickle("land_polygons.pkl")
    print("Land polygons saved to land_polygons.pkl")


    # Create a NetCDFLoader instance
    file_path = "data/raw/ozone.nc"  # Path to your NetCDF file
    loader = NetCDFLoader(file_path, crop_box=(lat_min, lat_max, lon_min, lon_max))

    #time of data
    print("Available variables:", loader.get_data_variable_names())
    variable_name = loader.get_data_variable_names()[0]
    #values grided
    values_grid = loader.get_grid_data(variable_name)[0]
    lat_grid, lon_grid = loader.get_grid_data(variable_name)[1:3]

    # Convert lat/lon to x/y projection
    x_grid, y_grid = transformer.latlon_to_xy(lon_grid, lat_grid)

    # Create a DataFrame with the projected data
    df_data = pd.DataFrame({
        "x": x_grid.flatten(),
        "y": y_grid.flatten(),
        "value": values_grid.flatten()
    })

    # Save the projected data to a CSV file
    df_data.to_csv("projected_data.csv", index=False)
    print("Projected data saved to projected_data.csv")

    # Plot the projected land polygons and data
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_grid, y_grid, values_grid, cmap='coolwarm', alpha=0.5)
    plt.colorbar(label=variable_name)   
    # Plot land polygons
    for x, y in land_polygons:
        plt.plot(x, y, color='black', linewidth=2)  # or fill with ax.fill(x, y, color='gray')
    plt.title("Projected Land Polygons and Data")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.savefig("projected_land_and_data.png")
    plt.show()
    print("Projected land polygons and data saved to projected_land_and_ice_temperature data.png")

if __name__ == "__main__":
    generate_projected_land_and_data()