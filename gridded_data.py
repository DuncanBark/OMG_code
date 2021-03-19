import os
import argparse
import numpy as np
import xarray as xr
import netCDF4 as nc4
from pathlib import Path
from datetime import datetime

# Return the nanmin, and nanmax of a list
# Checks to see if the data is all nans, if so it returns a nan for the min and max
# Primarily used to avoid taking the min/max of all NaN lists, which is common
def get_data_min_max(data):
    non_nans = np.count_nonzero(~np.isnan(data))
    if non_nans == 0:
        return [np.nan, np.nan]
    return [np.nanmin(data), np.nanmax(data)]

# Define fill values for binary and netcdf
array_precision = np.float32
if array_precision == np.float32:
    binary_output_dtype = '>f4'
    netcdf_fill_value = nc4.default_fillvals['f4']
elif array_precision == np.float64:
    binary_output_dtype = '>f8'
    netcdf_fill_value = nc4.default_fillvals['f8']

# Location to save dive netCDFs
netCDF_path = Path(f'{Path(__file__).resolve().parents[1]}/littleproby_emails/netCDFs')

# List of valid id_nums (probes that have produced valid netCDF files)
id_nums = os.listdir(netCDF_path)

# Pressure (height) list
# 0-1200dBar, 1dBar spacing
p_list = range(0, 1201, 1)

def gridded_data():
    for id_num in id_nums:
        # list of dive netcdfs for this probe id
        dive_netcdfs = list(netCDF_path.glob(f'**/{id_num}/*.nc'))

        # gridded path
        gridded_path = Path(f'{netCDF_path}/{id_num}/gridded/')

        for dive in dive_netcdfs:
            dive_ds = xr.open_dataset(dive)
            dn = dive_ds.attrs['dive_number']
            sn = dive_ds.attrs['probe_id']
            ascent_start_time = dive_ds['Code_rise_time'][0].values[0]
            ascent_end_time = dive_ds['Code_rise_time'][0].values[-1]

            # deltas takes the time values and calculates time delta from the ascent_start_time
            # time_s is the timedelta as seconds
            # time value list is flipped so that it is increasing in the final list
            deltas = [(x - ascent_start_time) for x in np.flip(dive_ds['Code_rise_time'][0].values)]
            time_ascent_s = [x.astype('timedelta64[s]').astype(int) for x in deltas]

            # Note: x values of the data points need to be increasing (accoding to numpy.interp) so flip reverses it
            # Need to also flip the data to match with the x values
            interp_T_asc = np.interp(p_list, np.flip(dive_ds['pressure_ascent'][0].values), np.flip(dive_ds['T_asc'][0].values), left=np.nan, right=np.nan)
            interp_S_asc = np.interp(p_list, np.flip(dive_ds['pressure_ascent'][0].values), np.flip(dive_ds['S_asc'][0].values), left=np.nan, right=np.nan)
            interp_TC_asc = np.interp(p_list, np.flip(dive_ds['pressure_ascent'][0].values), np.flip(dive_ds['TC_asc'][0].values), left=np.nan, right=np.nan)
            interp_time_asc = np.interp(p_list, np.flip(dive_ds['Code_rise_pressure'][0].values), time_ascent_s, left=np.nan, right=np.nan)

            # temperature data array
            asc_coords = {'ascent_end_time':[np.datetime64(dive_ds.attrs['time_coverage_end'][:-1])], 'pressure':(('pressure'), p_list)}
            dims = ['ascent_end_time', 'pressure']
            da_T_asc = np.expand_dims(interp_T_asc, 0)
            T_asc_min_max = get_data_min_max(interp_T_asc)
            new_T_asc_attrs = {'valid_min':T_asc_min_max[0], 'valid_max':T_asc_min_max[1]} # new valid_min and valid_max values from interpolated data
            new_T_asc_attrs = {key: new_T_asc_attrs.get(key, dive_ds['T_asc'].attrs[key]) for key in dive_ds['T_asc'].attrs} # updates previous attributes (from L2 data) to new interpolated values
            da_T_asc = xr.DataArray(da_T_asc, dims=dims, coords=asc_coords, attrs=new_T_asc_attrs)
            da_T_asc.name = 'T_asc'

            # salinity data array
            da_S_asc = np.expand_dims(interp_S_asc, 0)
            S_asc_min_max = get_data_min_max(interp_S_asc)
            new_S_asc_attrs = {'valid_min':S_asc_min_max[0], 'valid_max':S_asc_min_max[1]} # new valid_min and valid_max values from interpolated data
            new_S_asc_attrs = {key: new_S_asc_attrs.get(key, dive_ds['S_asc'].attrs[key]) for key in dive_ds['S_asc'].attrs} # updates previous attributes (from L2 data) to new interpolated values
            da_S_asc = xr.DataArray(da_S_asc, dims=dims, coords=asc_coords, attrs=new_S_asc_attrs)
            da_S_asc.name = 'S_asc'

            # tcond data array
            da_TC_asc = np.expand_dims(interp_TC_asc, 0)
            TC_asc_min_max = get_data_min_max(interp_TC_asc)
            new_TC_asc_attrs = {'valid_min':TC_asc_min_max[0], 'valid_max':TC_asc_min_max[1]} # new valid_min and valid_max values from interpolated data
            new_TC_asc_attrs = {key: new_TC_asc_attrs.get(key, dive_ds['TC_asc'].attrs[key]) for key in dive_ds['TC_asc'].attrs} # updates previous attributes (from L2 data) to new interpolated values
            da_TC_asc = xr.DataArray(da_TC_asc, dims=dims, coords=asc_coords, attrs=new_TC_asc_attrs)
            da_TC_asc.name = 'TC_asc'

            # time data array
            da_time_asc = np.expand_dims(interp_time_asc, 0)
            time_attrs = {'long_name':'ascent time', 'standard_name':'', 'coverage_content_type':'coordinate', 'axis':'T'}
            da_time_asc = xr.DataArray(da_time_asc, dims=dims, coords=asc_coords, attrs=time_attrs)
            da_time_asc.name = 'time_asc'

            # Merge all data arrays into one dataset
            ds = xr.merge([da_T_asc, da_S_asc, da_TC_asc, da_time_asc])

            # Pressure coord attributes
            ds['pressure'].attrs = {'long_name':'pressure', 'standard_name':'', 'units':'dBar', 'coverage_content_type':'physicalMeasurement', 'valid_min':np.nanmin(p_list), 'valid_max':np.nanmax(p_list)}

            # Dataset Metadata
            ds_attrs = {}
            ds_attrs['probe_id'] = sn
            ds_attrs['dive_number'] = dn
            ds_attrs['title'] = 'OMG Ocean ALAMO CTD Level 3 Data'
            ds_attrs['summary'] = 'This file contains salinity, temperature, tcond, and pressure measurements from air-deployed autonomous probes. This product is gridded on pressure, with a resolution of 1 dBar, a minimum value of 0 dBar, and a maximum value of 1200 dBar.'
            ds_attrs['keywords'] = 'Water Temperature, Salinity'
            ds_attrs['keywords_vocabulary'] = 'NASA Global Change Master Directory (GCMD) Science Keywords'
            ds_attrs['Conventions'] = ''
            ds_attrs['id'] = ''
            ds_attrs['uuid'] = ''
            ds_attrs['naming_authority'] = 'gov.nasa.jpl'
            ds_attrs['cdm_data_type'] = ''
            ds_attrs['featureType'] = ''
            ds_attrs['history'] = 'Takes previously made dive specific L2 NetCDF data files and interpolates the data to a regular pressure grid and saves the result to a NetCDF file.'
            ds_attrs['source'] = 'Dive specific L2 NetCDF files'
            ds_attrs['platform'] = ''
            ds_attrs['instrument'] = ''
            ds_attrs['processing_level'] = 'L3'
            ds_attrs['comment'] = 'The data was collected by a single probe over a specific dive.'
            ds_attrs['standard_name_vocabulary'] = 'NetCDF Climate and Forecast (CF) Metadata Convention'
            ds_attrs['acknowledgement'] = 'This research was carried out by the Jet Propulsion Laboratory, managed by the California Institute of Technology under a contract with the National Aeronautics and Space Administration.'
            ds_attrs['license'] = 'Public Domain'
            ds_attrs['product_version'] = '1.0'
            ds_attrs['references'] = ''
            ds_attrs['creator_name'] = 'OMG Science Team'
            ds_attrs['creator_email'] = 'omg-science@jpl.nasa.gov'
            ds_attrs['creator_url'] = 'https://dx.doi.org/10.5067/OMGEV-CTDS1'
            ds_attrs['creator_type'] = 'group'
            ds_attrs['creator_institution'] = 'NASA Jet Propulsion Laboratory'
            ds_attrs['institution'] = 'NASA Jet Propulsion Laboratory'
            ds_attrs['project'] = 'Oceans Melting Greenland (OMG)'
            ds_attrs['program'] = ''
            ds_attrs['contributor_name'] = ''
            ds_attrs['contributor_role'] = ''
            ds_attrs['publisher_name'] = 'PO.DAAC'
            ds_attrs['publisher_email'] = 'podaac@podaac.jpl.nasa.gov'
            ds_attrs['publisher_url'] = 'https://dx.doi.org/10.5067/OMGEV-CTDS1'
            ds_attrs['publisher_type'] = 'group'
            ds_attrs['publisher_institution'] = 'NASA Jet Propulsion Laboratory'
            ds_attrs['geospatial_lat_min'] = dive_ds.attrs['gps_dive_end_lat']
            ds_attrs['geospatial_lat_max'] = dive_ds.attrs['gps_dive_end_lat']
            ds_attrs['geospatial_lat_units'] = 'degrees_north'
            ds_attrs['geospatial_lat_resolution'] = 1.0E-7
            ds_attrs['geospatial_lon_min'] = dive_ds.attrs['gps_dive_end_lon']
            ds_attrs['geospatial_lon_max'] = dive_ds.attrs['gps_dive_end_lon']
            ds_attrs['geospatial_lon_units'] = 'degrees_east'
            ds_attrs['geospatial_lon_resolution'] = 1.0E-7
            ds_attrs['geospatial_vertical_min'] = np.nanmin(p_list)
            ds_attrs['geospatial_vertical_max'] = np.nanmax(p_list)
            ds_attrs['geospatial_vertical_resolution'] = '1'
            ds_attrs['geospatial_vertical_units'] = 'dBar'
            ds_attrs['geospatial_vertical_positive'] = 'down'
            ds_attrs['time_coverage_start'] = str(np.datetime64(ascent_start_time, 's'))
            ds_attrs['time_coverage_end'] = str(np.datetime64(ascent_end_time, 's'))
            ds_attrs['time_coverage_duration'] = str(np.datetime64(ascent_end_time, 's') - np.datetime64(ascent_start_time, 's'))
            ds_attrs['date_created'] = str(np.datetime64('now'))
            ds_attrs['source_product_header'] = ''
            for old_attr in dive_ds.attrs:
                if 'param_' in old_attr:
                    ds_attrs[old_attr] = dive_ds.attrs[old_attr]

            # Assign attributes to dataset
            ds.attrs = ds_attrs

            # Create directory if not present
            if not os.path.exists(gridded_path):
                os.makedirs(gridded_path)

            # loop through data attributes
            # check if int32 or int64, encode it as int16
            # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH DATA VAR
            dv_encoding = dict()
            for dv in ds.data_vars:
                dv_encoding[dv] =  {'zlib':True, \
                                    'complevel':5,\
                                    'shuffle':True,\
                                    '_FillValue':netcdf_fill_value}
            # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH COORDINATE
            coord_encoding = dict()
            for coord in ds.coords:
                # default encoding: no fill value, float32
                coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
                if (ds[coord].values.dtype == np.int32) or \
                    (ds[coord].values.dtype == np.int64) :
                    coord_encoding[coord]['dtype'] ='int32'
                if 'time' in coord:
                    coord_encoding[coord]['dtype'] ='int32'
            # MERGE ENCODINGS for coordinates and variables
            encoding = {**dv_encoding, **coord_encoding}

            # Save dataset with the name "XXXX_dive_NNNN_dataset_L3.nc" where XXXX is the probe id and NNNN is the dive number
            ds.to_netcdf(path=f'{gridded_path}/{sn}_dive_{str(dn).zfill(4)}_dataset_L3.nc', encoding=encoding)

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--netcdf_path', default=False, nargs=1,
                        help='grabs from provided directory and outputs created netcdf files to provided directory')


    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.netcdf_path:
        print(f'Using netcdf_path path: {args.netcdf_path[0]}')
        netCDF_path = Path(args.netcdf_path[0])
    else:
        print(f'Using default netcdf_path path: {netCDF_path}')

    gridded_data()