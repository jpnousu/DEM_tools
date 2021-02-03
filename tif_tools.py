## AutoGIS 2020 final assignment
## Tools to do simple hydrological dem preprocessing
## These functions still need a lot of testing


import os
import urllib
import rasterio
import numpy as np
from numpy import newaxis
from rasterio.plot import show
import matplotlib.pyplot as plt
import glob
from pyproj import CRS
from rasterio.merge import merge
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg


def raster_from_mml(dirpath, subset, layer='korkeusmalli_2m', form='image/tiff', plot=False, cmap='terrain'):
    
    '''Downloads a raster from MML database and writes it to dirpath folder in local memory
        
        Parameters:
        subset = boundary coordinates [minx, miny, maxx, maxy] (list)
        layer = the layer wanted to fetch e.g. 'korkeusmalli_2m' or 'korkeusmalli_10m' (str)
        form = form of the raster e.g 'image/tiff' (str)
        plot = whether or not to plot the created raster, True/False
        cmap = colormap for plotting (str - default = 'terrain')
        '''

    # The base url for maanmittauslaitos
    url = 'https://beta-karttakuva.maanmittauslaitos.fi/wcs/service/ows?'

    # Defining the latter url code
    params = dict(service='service=WCS', 
                  version='version=2.0.1', 
                  request='request=GetCoverage', 
                  CoverageID=f'CoverageID={layer}', 
                  SUBSET=f'SUBSET=E({subset[0]},{subset[2]})&SUBSET=N({subset[1]},{subset[3]})',
                  outformat=f'format={form}',
                  compression='geotiff:compression=LZW')
    
    par_url = ''
    for par in params.keys():
        par_url += params[par] + '&'
    par_url = par_url[0:-1]
    new_url = (url + par_url)
    
    # Putting the whole url together
    r = urllib.request.urlretrieve(new_url)
    
    # Open the file with the url:
    raster = rasterio.open(r[0])
    
    del r
        
    out_fp = os.path.join(dirpath, layer) + '.tif'
    
    # Copy the metadata
    out_meta = raster.meta.copy()
    
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": raster.height,
                     "width": raster.width,
                     "transform": raster.meta['transform'],
                     "crs": raster.meta['crs']
                         }
                    )
    
    # Manipulating the data for writing purpose
    raster_dem = raster.read(1)
    raster_dem = raster_dem[newaxis, :, :]

    # Write the raster to disk
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(raster_dem)
        
    # Open the raster file
    raster_dem = rasterio.open(out_fp)
        
    # Plot the result
    if plot == True:
        fig, (ax1) = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        img = show(raster_dem, ax=ax1, cmap=cmap)
        img.set_title('Fetched raster from National Land Survey of Finland')

        
    
    return raster_dem

def show_files(dirpath, search_criteria='*.tif', cmap='terrain'):
    
    ''' This function plots the first four tif-files from local dirpath directory
        Can be used to test if the tif files look correct and ready for further analysis
        
        Parameters:
        dirpath = the local repository where the tif files are (str)
        search_criteria = criteria to select files (str)
        cmap = colormap for plotting (str - default = 'terrain')

        '''
    
    # select the files
    q = os.path.join(dirpath, search_criteria)

    # glob function can be used to list files from a directory with specific criteria
    dem_fps = glob.glob(q)
    
    # List for the source files
    src_files_to_mosaic = []

    # Iterate over raster files and add them to source -list in 'read mode'
    for fp in dem_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    file_len = len(src_files_to_mosaic)
    
    # Create 4 plots next to each other
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(24, 6))

    # Plot first four files
    show(src_files_to_mosaic[0], ax=ax1, cmap=cmap)
    show(src_files_to_mosaic[1], ax=ax2, cmap=cmap)
    show(src_files_to_mosaic[2], ax=ax3, cmap=cmap)
    show(src_files_to_mosaic[3], ax=ax4, cmap=cmap)
    fig.suptitle('The first four tif files plotted below')
    plt.tight_layout()
    plt.show()


def tifs_to_mosaic(dirpath, out_fn, search_criteria='*.tif', plot=False, cmap='terrain'):

    ''' This function reads multiple tif-files from local directory, makes a mosaic of them and saves the mosaic as a tif
        Can be used e.g. to construct a single DEM file from manually downloaded tif files from National Land Survey of Finland database
        
        Parameters:
        dirpath = the local repository where the tif files are (str)
        out_fn = output filename (str)
        plot = whether or not to plot the created raster, True/False
        cmap = colormap for plotting (str - default = 'terrain')
'''
    

    # Creates a new folder for processed rasters
    if not os.path.exists(f'{dirpath}/processed'):
        os.makedirs(f'{dirpath}/processed')
    
    out_fp = os.path.join(dirpath, 'processed', out_fn)

    # select the files
    q = os.path.join(dirpath, search_criteria)

    # glob function can be used to list files from a directory with specific criteria
    dem_fps = glob.glob(q)
    
    # List for the source files
    src_files_to_mosaic = []

    # Iterate over raster files and add them to source -list in 'read mode'
    for fp in dem_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
            
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Copy the metadata
    out_meta = src.meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": CRS.from_epsg(3067)
                     }
                    )

    # Write the mosaic raster to disk
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Open the clipped raster file
    mosaic = rasterio.open(out_fp)
    
    # Plot the result
    if plot == True:
        fig, (ax1) = plt.subplots(ncols=1, nrows=1, figsize=(12, 4))
        img = show(mosaic, ax=ax1, cmap=cmap)
        img.set_title('Mosaic raster')
        
    return mosaic
        
def tif_clip(subset, dirpath, in_fn, out_fn, plot=False, cmap='terrain'):

    '''Clips a tif raster with user defined coordinate boundary box.
        
        Parameters 
        subset = coordinates [minx, miny, maxx, maxy] (list)
        dirpath = the local repository where the tif file is (str)
        in_fn = input filename (str)
        out_fn = output filename (str)
        plot = whether or not to plot the created raster, True/False
        cmap = colormap for plotting (str - default = 'terrain')
        
        To do: there is a warning, fix the code

        '''
    

    # Input raster 
    fp = os.path.join(dirpath, in_fn)
    
    # Output raster
    
    # File and folder paths
    if not dirpath == 'data/processed':
        if not os.path.exists(f'{dirpath}/processed'):
            os.makedirs(f'{dirpath}/processed')

    if dirpath == 'data':
        out_tif = os.path.join(dirpath, 'processed', out_fn)
    else:
        out_tif = os.path.join(dirpath, out_fn)
        
    
    # Read the data
    data = rasterio.open(fp)

    bbox = box(subset[0], subset[1], subset[2], subset[3])
    
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(3067))
    
    # Project the Polygon into same CRS as the grid 
    geo = geo.to_crs(crs=data.crs)
    
    def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
        import json
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

    coords = getFeatures(geo)
    #print(coords)
    
    # Clip the raster with Polygon
    out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)
    
    # Copy the metadata
    out_meta = data.meta.copy()
    #print(out_meta)
    
    # Parse EPSG code
    epsg_code = int(data.crs.data['init'][5:])
    #print(epsg_code)
    
    # Updating the metadata
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     "crs": data.meta['crs']}
                             )
    with rasterio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)
            
    # Open the clipped raster file
    clipped = rasterio.open(out_tif)

#    if plot == True:
#        # Visualize
#        fig, (ax1) = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
#        img = show(clipped, ax=ax1, cmap='terrain')

    if plot == True:
        # Visualize
        fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(12, 12))
        img2 = show(data, ax=ax1, cmap=cmap)
        geo.plot(ax=ax1, linewidth=0.5, edgecolor='red', alpha=0.7)
        img = show(clipped, ax=ax2, cmap=cmap)
        img2.set_title('Original raster with subset box')
        img.set_title('Clipped raster')
        plt.tight_layout()

    return clipped
        
        
def d8_flowdir(in_fn, dirpath='data', out_fn='d8_flowdir.tif', plot=True, cmap='terrain'):
    
    '''Takes a DEM and defines flow direction for each cell and returns a raster of flow directions in degrees radius. -1 means there is no flow
    
    Parameters:
    in_fn = input filename (str)
    dirpath = the local repository where the tif files are (str)
    out_fn = output filename (str)
    plot = whether or not to plot the created raster, True/False
    cmap = colormap for plotting (str - default = 'terrain')

    Note: Change it to use numpy element-wise operations to be quicker
    '''


    fp = os.path.join(dirpath, in_fn)  
    
        # File and folder paths
    if not dirpath == 'data/processed':
        if not os.path.exists(f'{dirpath}/processed'):
            os.makedirs(f'{dirpath}/processed')

    if dirpath == 'data':
        out_fp = os.path.join(dirpath, 'processed', out_fn)
    else:
        out_fp = os.path.join(dirpath, out_fn)
        
        
    # Read the data
    data = rasterio.open(fp)
    dem = data.read(1)
    
    # create dem.max + 1 to avoid flow into NA cells
    max = dem.max() + 1
    
    # new array for flow direction raster
    flow_dir = dem.copy()
    
    # directions used:
    # -1 = no flow (pit)
    # otherwise degrees radius
    
    degrees = [-1, 135, 90, 45, 0, 315, 270, 225, 180]
    
    # loop over the whole np array 
    for r in range(dem.shape[0]):
        for c in range(dem.shape[1]):
            cell = dem[r,c]
            if cell != -9999:
                # when cell not first or last row/column
                if r > 0 and r < dem.shape[0]-1 and c > 0 and c < dem.shape[1]-1:
                    around = [cell, dem[r-1, c-1], dem[r-1, c], dem[r-1, c+1], dem[r, c+1], dem[r+1, c+1], dem[r+1, c], dem[r+1, c-1], dem[r, c-1]]
                
                # when cell first row and column
                if r == 0 and c == 0:
                    around = [cell, max, max, max, dem[r, c+1], dem[r+1, c+1], dem[r+1, c], max, max]
                
                # when cell last row and column
                if r == dem.shape[0]-1 and c == dem.shape[1]-1:
                    around = [cell, dem[r-1, c-1], dem[r-1, c], max, max, max, max, max, dem[r, c-1]]
                
                # when cell first row and column is not first or last
                if r == 0 and c > 0 and c < dem.shape[1]-1:
                    around = [cell, max, max, max, dem[r, c+1], dem[r+1, c+1], dem[r+1, c], dem[r+1, c-1], dem[r, c-1]]
                
                # when cell last row and column is not first or last
                if r == dem.shape[0]-1 and c > 0 and c < dem.shape[1]-1:
                    around = [cell, dem[r-1, c-1], dem[r-1, c], dem[r-1, c+1], dem[r, c+1], max, max, max, dem[r, c-1]]
                
                # when cell first column and row is not first or last
                if r > 0 and r < dem.shape[0]-1 and c == 0:
                    around = [cell, max, dem[r-1, c], dem[r-1, c+1], dem[r, c+1], dem[r+1, c+1], dem[r+1, c], max, max]
                
                # when cell last column and row is not first or last
                if r > 0 and r < dem.shape[0]-1 and c == dem.shape[1]-1:
                    around = [cell, dem[r-1, c-1], dem[r-1, c], max, max, max, dem[r+1, c], dem[r+1, c-1], dem[r, c-1]]
            
                # in case NA (=-9999) in the constructed around list -> assigned to max
                if -9999 in around:
                    around[around.index(-9999)] = max
            
            # finding out the degrees radius by using the index of minimum cell and previously created degrees list
            min_index = degrees[around.index(min(around))]
            flow_dir[r,c] = min_index
            
            #if cell == -9999:
            #    flow_dir[r,c] = dem[r,c]
        
    # condition to plot    
    if plot == True:
        fig, ax = plt.subplots(figsize=(12, 12))
        img = ax.imshow(flow_dir, cmap=cmap)
        ax.set_title('Flow directions as degrees radius')
        fig.colorbar(img)
        fig = plt.figure()
        ax = fig.add_subplot(polar=True)
        ax.set_yticklabels([])
        ax.set_title('Degrees radius')


    out_meta = data.meta.copy()
    
    # Manipulating the data for writing purpose
    flow_dir = flow_dir[newaxis, :, :]


    #The saving does not work yet
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(flow_dir)
        
    flow_dir = rasterio.open(out_fp)
    
    
    return flow_dir