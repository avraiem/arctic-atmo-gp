from abc import ABC, abstractmethod
import xarray as xr

class BaseNetCDFLoader(ABC):
    def __init__(self, file_path, crop_box=None):
        self.file_path = file_path
        self.ds = xr.open_dataset(file_path)
        self.crop_box = crop_box

    def get_variable_names(self):
        return list(self.ds.variables)

    def get_variable_info(self, variable_name):
        var = self.ds[variable_name]
        return {
            "name": variable_name,
            "dimensions": var.dims,
            "shape": var.shape,
            "dtype": str(var.dtype),
            "attrs": var.attrs
        }

    @abstractmethod
    def get_lat_lon_names(self):
        pass

    @abstractmethod
    def get_lat_lon_values(self, crop_box=None):
        pass

    def close(self):
        self.ds.close()