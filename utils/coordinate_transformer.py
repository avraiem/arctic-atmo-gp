from pyproj import Transformer
import numpy as np
import pandas as pd
from cartopy.feature import NaturalEarthFeature
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import box

class CoordinateTransformer:
    def __init__(self, from_crs="epsg:4326", to_crs="epsg:3413"):
        """
        Handles coordinate transformations between geographic (lat/lon) and projected (x/y).
        Default CRS: WGS84 → NSIDC Sea Ice Polar Stereographic North (EPSG:3413).
        """
        self.transformer_to = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        self.transformer_from = Transformer.from_crs(to_crs, from_crs, always_xy=True)

    def latlon_to_xy(self, lon, lat):
        """
        Convert lon/lat to x/y (in meters).
        Accepts scalars, 1D arrays, or 2D grids.
        """
        x, y = self.transformer_to.transform(lon, lat)
        return x, y

    def xy_to_latlon(self, x, y):
        """
        Convert x/y back to lon/lat.
        """
        lon, lat = self.transformer_from.transform(x, y)
        return lon, lat

    def grid_to_xy(self, lon_grid, lat_grid):
        """
        Convert 2D lon/lat grids to 2D x/y grids.
        """
        x, y = self.latlon_to_xy(lon_grid, lat_grid)
        return np.array(x), np.array(y)

    def flatten_grid_to_xy(self, lon_grid, lat_grid):
        """
        Convert 2D grids to flat arrays (N, 2) for GP input.
        """
        x, y = self.grid_to_xy(lon_grid, lat_grid)
        return np.stack([x.ravel(), y.ravel()], axis=1)

    def flatten_grid_to_latlon(self, x_grid, y_grid):
        """
        Convert 2D x/y grids back to flat lon/lat coordinates.
        """
        lon, lat = self.xy_to_latlon(x_grid, y_grid)
        return np.stack([lon.ravel(), lat.ravel()], axis=1)


    def get_flat_dataframe(self, lon_grid, lat_grid, values, to_xy=True):
        """
        Converts 2D lat/lon and value grids into a flat DataFrame with either:
            - lat, lon, value
            - x, y, value  (if to_xy=True)
        """
        if to_xy:
            x, y = self.latlon_to_xy(lon_grid, lat_grid)
            x_flat = x.ravel()
            y_flat = y.ravel()
            df = pd.DataFrame({
                "x": x_flat,
                "y": y_flat,
                "value": values.ravel()
            })
        else:
            df = pd.DataFrame({
                "lat": lat_grid.ravel(),
                "lon": lon_grid.ravel(),
                "value": values.ravel()
            })
        return df

    def get_grid_dataframe(self, x_grid, y_grid, values):
        """
        Returns a DataFrame indexed by x and y for gridded data.
        x and y are assumed to be 1D arrays (grid axes).
        """
        if x_grid.ndim != 2 or y_grid.ndim != 2:
            raise ValueError("x_grid and y_grid must be 2D.")

        return pd.DataFrame(data=values, index=x_grid[:, 0], columns=y_grid[0])
    
    def flat_latlon_to_xy_dataframe(self, df_latlon: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a flat DataFrame with columns ['lat', 'lon', 'value']
        to ['x', 'y', 'value'] using the projection.

        Parameters:
            df_latlon (pd.DataFrame): Must contain 'lat', 'lon', 'value' columns

        Returns:
            pd.DataFrame: with 'x', 'y', 'value'
        """
        if not {"lat", "lon", "value"}.issubset(df_latlon.columns):
            raise ValueError("Input DataFrame must contain columns: 'lat', 'lon', 'value'")

        lon = df_latlon["lon"].values
        lat = df_latlon["lat"].values
        val = df_latlon["value"].values

        x, y = self.latlon_to_xy(lon, lat)

        return pd.DataFrame({
            "x": x,
            "y": y,
            "value": val
        })
    
    def flat_xy_to_latlon_dataframe(self, df_xy: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a flat DataFrame with columns ['x', 'y', 'value']  to ['lat', 'lon', 'value'] using the inverse projection.

        Parameters:
            df_xy (pd.DataFrame): Must contain 'x', 'y', 'value' columns

        Returns:
            pd.DataFrame: with 'lat', 'lon', 'value'
        """
        if not {"x", "y", "value"}.issubset(df_xy.columns):
            raise ValueError("Input DataFrame must contain columns: 'x', 'y', 'value'")

        x = df_xy["x"].values
        y = df_xy["y"].values
        val = df_xy["value"].values

        lon, lat = self.xy_to_latlon(x, y)

        return pd.DataFrame({
            "lat": lat,
            "lon": lon,
            "value": val
        })
    
    def get_transformed_land_polygons(self, extent=None, resolution="110m", wrap_longitudes=True):
        """
        Returns a list of transformed land polygons in (x, y) space.
        
        Parameters:
            resolution (str): One of '110m', '50m', '10m'
            extent (tuple): Optional bounding box (min_lat, max_lat, min_lon, max_lon)
            wrap_longitudes (bool): Whether to convert longitudes >180 to [-180, 180]
        
        Returns:
            List of (x_coords, y_coords) tuples for plotting.
        """
        land = NaturalEarthFeature('physical', 'land', scale = resolution)

        #check if extent's longitudes are in [0, 360] and convert to [-180, 180] if needed
        if extent is not None:
            if wrap_longitudes:
                extent = (
                    extent[0], extent[1],
                    self.wrap_lon_to_180(extent[2]), self.wrap_lon_to_180(extent[3])
                )

        clip_box = box(extent[2], extent[0], extent[3], extent[1]) if extent is not None else None

        transformed_polys = []

        for geom in land.geometries():
            if clip_box:
                geom = geom.intersection(clip_box)
                if geom.is_empty:
                    continue

            if isinstance(geom, Polygon):
                coords = list(geom.exterior.coords)
                lon, lat = zip(*coords)
                x, y = self.latlon_to_xy(lon, lat)
                transformed_polys.append((x, y))

            elif isinstance(geom, MultiPolygon):
                for part in geom.geoms:
                    if clip_box:
                        part = part.intersection(clip_box)
                        if part.is_empty:
                            continue
                    coords = list(part.exterior.coords)
                    lon, lat = zip(*coords)
                    x, y = self.latlon_to_xy(lon, lat)
                    transformed_polys.append((x, y))

        return transformed_polys
    

    def get_transformed_feature_polygons(self, extent=None, resolution="10m", wrap_longitudes=True, include_minor_islands=True):
        """
        Returns a list of transformed land polygons in (x, y) space.

        Parameters:
            extent (tuple): (min_lat, max_lat, min_lon, max_lon) in degrees
            resolution (str): '10m', '50m', or '110m'
            wrap_longitudes (bool): Whether to convert longitudes >180 to [-180, 180]
            include_minor_islands (bool): Whether to also include minor islands

        Returns:
            List of (x_coords, y_coords) tuples for plotting.
        """

        # Prepare extent and clip box
        clip_box = None
        if extent is not None:
            if wrap_longitudes:
                extent = (
                    extent[0], extent[1],
                    self.wrap_lon_to_180(extent[2]), self.wrap_lon_to_180(extent[3])
                )
            clip_box = box(extent[2], extent[0], extent[3], extent[1])

        transformed_polys = []

        # List of features to extract
        features_to_process = [
            NaturalEarthFeature('physical', 'land', scale=resolution)
        ]
        if include_minor_islands:
            features_to_process.append(
                NaturalEarthFeature('physical', 'minor_islands', scale=resolution)
            )

        features_to_process.append(NaturalEarthFeature('physical', 'coastline', scale=resolution))
        features_to_process.append(NaturalEarthFeature('cultural', 'urban_areas', scale=resolution))

        # Process each feature
        for feature in features_to_process:
            for geom in feature.geometries():
                if clip_box:
                    geom = geom.intersection(clip_box)
                    if geom.is_empty:
                        continue

                if isinstance(geom, Polygon):
                    coords = list(geom.exterior.coords)
                    lon, lat = zip(*coords)
                    x, y = self.latlon_to_xy(lon, lat)
                    transformed_polys.append((x, y))

                elif isinstance(geom, MultiPolygon):
                    for part in geom.geoms:
                        if clip_box:
                            part = part.intersection(clip_box)
                            if part.is_empty:
                                continue
                        coords = list(part.exterior.coords)
                        lon, lat = zip(*coords)
                        x, y = self.latlon_to_xy(lon, lat)
                        transformed_polys.append((x, y))

        print(f"Total transformed land polygons extracted: {len(transformed_polys)}")
        return transformed_polys
    
    @staticmethod
    def wrap_lon_to_180(lon_array):
        """
        Converts longitudes from [0, 360] to [-180, 180] range.
        """
        return ((lon_array + 180) % 360) - 180

    @staticmethod
    def wrap_lon_to_360(lon_array):
        """
        Converts longitudes from [-180, 180] to [0, 360] range.
        """
        return lon_array % 360