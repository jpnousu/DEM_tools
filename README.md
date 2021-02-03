## Topic: 

Digital Elevation Model (DEM) preprocessing for hydrological analysis / spatial hydrological models

### Structure of this repository:

Main repository contains tif_tools.py module in which functions were created. Main repository also contains a notebook dem_preprocess_demo.ipynb which demonstrates the use of tif_tools functions.
Folder 'data' includes the manually downloaded DEM geotiff files from National Land Survey of Finland. When the code is run, another folder 'data/processed' is made were the processed rasters are saved.

### Input data:

Digital Elevation Models (DEM) from National Land Survey of Finland -> 'data' folder V*.tif files

### Analysis steps:

Most of the work was conducted in tif_tools.py module in which useful functions were created. These functions were constructed to automate the preprocessing of digital elevation model (DEM) for hydrological analysis and/or spatial hydrological modelling.
The functions were first created and tested with manually downloaded DEM tif rasters from National Land Survey of Finland. Furthermore, the module was advanced to be able to automate the download process from National Land Survey of Finland.

Notebook dem_preprocess_demo.ipynb demonstrates the use of tif_tools functions
These functions include:
- Plotting tif files from a local repository
- Creating a mosaic of multiple tif files 
- User defined clipping for tif
- Automated digital elevation model (DEM) download from National Land Survey of Finland database
- Defining flow direction for hydrological use

### Results:

- Useful functions to automate raster download and processing tools.
- Flow direction raster

### References:

- Literature related to the topic
- AutoGIS course materials related to raster processing (https://autogis-site.readthedocs.io/en/latest/lessons/Raster/overview.html)
- Rasterio documentation (https://rasterio.readthedocs.io/en/latest/)
