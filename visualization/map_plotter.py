import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils.coordinate_transformer import CoordinateTransformer
import numpy as np


class MapPlotter:
    def __init__(self, projection="polar"):
        if projection == "polar":
            self.crs = ccrs.NorthPolarStereo()
        else:
            self.crs = ccrs.PlateCarree()  # fallback for global or rectangular view

    def plot_grid(self, lon_grid, lat_grid, value_grid, title="Arctic Field", cmap="viridis", show=True, save_path=None):
        """
        Plots a gridded variable on a map using Cartopy.

        Parameters:
            lon_grid: 2D numpy array of longitudes
            lat_grid: 2D numpy array of latitudes
            value_grid: 2D array of values (same shape as lat/lon)
            title: plot title
            cmap: matplotlib colormap
            show: whether to display the plot
            save_path: optional path to save the image
        """
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=self.crs)

        mesh = ax.pcolormesh(lon_grid, lat_grid, value_grid,
                             transform=ccrs.PlateCarree(), cmap=cmap)
        plt.colorbar(mesh, orientation='vertical', pad=0.05, aspect=30)
        ax.set_title(title)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        
        plt.close(fig)

    def plot_points(self, lon, lat, values, title="Arctic Points", cmap="viridis", show=True, save_path=None):
        """
        Plots points on a map using Cartopy.

        Parameters:
            lon: 1D numpy array of longitudes
            lat: 1D numpy array of latitudes
            values: 1D array of values corresponding to the points
            title: plot title
            cmap: matplotlib colormap
            show: whether to display the plot
            save_path: optional path to save the image
        """
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=self.crs)

        scatter = ax.scatter(lon, lat, c=values, cmap=cmap,
                             transform=ccrs.PlateCarree(), s=10)
        plt.colorbar(scatter, orientation='vertical', pad=0.05, aspect=30)
        ax.set_title(title)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        
        plt.close(fig)


    def plot_arctic_field(self, lat, lon, values, title="Arctic Field", cmap="coolwarm", save_path=None, show=True, extent=None):
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        if extent is None:   
            ax.set_extent([-120, -50, 65, 85], crs=ccrs.PlateCarree())
        else:
            print("Using custom extent:", extent)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.gridlines(draw_labels=True)

        mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmap)
        plt.colorbar(mesh, orientation='vertical', pad=0.05, aspect=30, label="Value")
        ax.set_title(title)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)


    def plot_field(self, lat, lon, values=None, title="Field Plot", cmap="coolwarm", show=True, save_path=None, extent=None):
        """
        Plots a scalar field or point values on a map using Cartopy.
        Automatically selects projection based on latitude range.

        Parameters:
            lat, lon: 1D or 2D numpy arrays of coordinates (in degrees)
            values: 1D or 2D array of scalar field (optional)
            title: Title of the plot
            cmap: Colormap name
            show: Whether to display the figure
            save_path: If set, saves the figure to this path
            extent: Optional extent [lon_min, lon_max, lat_min, lat_max]
        """
        # Determine if we're plotting Arctic data
        is_arctic = np.mean(lat) > 60

        # Choose projection
        crs_proj = ccrs.NorthPolarStereo() if is_arctic else ccrs.PlateCarree()
        crs_data = ccrs.PlateCarree()

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=crs_proj)

        # Set extent
        if extent is None:
            if is_arctic:
                extent = [-120, -50, 65, 85]
            else:
                margin = 1.0
                lon_min, lon_max = np.min(lon), np.max(lon)
                lat_min, lat_max = np.min(lat), np.max(lat)
                extent = [lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin]
        ax.set_extent(extent, crs=crs_data)

        # Add features
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor='black', zorder=1)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), zorder=0)
        ax.gridlines(draw_labels=True)

        # Plot data
        if values is not None:
            mesh = ax.pcolormesh(lon, lat, values, transform=crs_data, cmap=cmap)
            plt.colorbar(mesh, orientation='vertical', pad=0.05, aspect=30, label="Value")
        else:
            ax.scatter(lon, lat, s=5, c='red', transform=crs_data)

        ax.set_title(title)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_transformed_overlay_on_map(x_grid, y_grid, values=None, title="Transformed Grid Overlay", cmap="viridis"):
        """
        Visualizes x/y data by transforming it back to lat/lon and overlaying on a map.

        Arguments:
            x_grid (2D np.ndarray): Projected X coordinates (meters)
            y_grid (2D np.ndarray): Projected Y coordinates (meters)
            values (2D np.ndarray or None): Optional scalar field (e.g., ice thickness)
            title (str): Title of the plot
            cmap (str): Matplotlib colormap for data

        Displays:
            Cartopy map with coastlines and overlaid data
        """
        transformer = CoordinateTransformer()

        # Back-project to geographic space
        lon_grid, lat_grid = transformer.xy_to_latlon(x_grid, y_grid)

        # Setup polar stereographic map
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        ax.set_extent([-120, -50, 65, 85], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.LAND, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.gridlines(draw_labels=True)

        if values is not None:
            mesh = ax.pcolormesh(lon_grid, lat_grid, values, transform=ccrs.PlateCarree(), cmap=cmap)
            plt.colorbar(mesh, orientation='vertical', pad=0.05, label="Value")
        else:
            ax.scatter(lon_grid, lat_grid, s=1, color='red', transform=ccrs.PlateCarree(), zorder=2)

        ax.set_title(title)
        plt.show()