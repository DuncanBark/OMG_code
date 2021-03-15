import os
import csv
import json
import numpy as np
import xarray as xr
import netCDF4 as nc4
from pathlib import Path
from collections import defaultdict

# Define fill values for binary and netcdf
array_precision = np.float32
if array_precision == np.float32:
    binary_output_dtype = '>f4'
    netcdf_fill_value = nc4.default_fillvals['f4']
elif array_precision == np.float64:
    binary_output_dtype = '>f8'
    netcdf_fill_value = nc4.default_fillvals['f8']

# Dim is the actual dimension list of values
# Dim len is length of the dimension. Only applies if data is longer than the dimension
# extra is tuple with sn and dn: (sn, dn)
def make_DA(data, dims, coords, attrs, dim_data, extra):
    dim_len = len(dim_data)
    sn, dn = extra
    if len(data[0])-dim_len >= 20:
        # Prints out statement if there are over 20 values beyond available pressure values
        # Checks if the last pressure value is less than 20 meaning probe is near the surface
        if dim_data[-1] < 20 and 'Parked' in attrs["long_name"]:
            print(f'Probe {sn}, dive {dn}, {attrs["long_name"]} (near surface) has more than 20 values ({len(data[0])-dim_len} values) beyond available pressure values')
        elif 'Parked' in attrs["long_name"]:
            print(f'Probe {sn}, dive {dn}, {attrs["long_name"]} (at depth with pressure ~{dim_data[-1]}) has more than 20 values ({len(data[0])-dim_len} values) beyond available pressure values')
        else:
            print(f'Probe {sn}, dive {dn}, {attrs["long_name"]} has more than 20 values ({len(data[0])-dim_len} values) beyond available pressure values')
    da = xr.DataArray(data=[data[0][:dim_len]],
                    dims=dims,
                    coords=coords,
                    attrs=attrs)
    return da

# Converts numerical array time_in_secs to a numpy array of datetime64 objects using a reference_time
def sec_to_dt64(ref_time, time_in_secs):
    new_times = []
    for time_in_sec in time_in_secs:
        new_times.append(ref_time + np.timedelta64(time_in_sec, 's'))
    return np.array(new_times)

# Gets scientific data from JSONs
# Ex location = dive['science']['ascending']['binned']
# Ex sci_names = ['pressure', 'temperature', 'salinity', 'tcond']
def get_data(location, sci_names):
    ret_list = []
    max_data_length = 0
    # Get maximum data length to make empty data an equal length of nans
    for sci_name in sci_names:
        if sci_name in location.keys():
            if len(location[sci_name]) > max_data_length:
                max_data_length = len(location[sci_name])
    # Get data from the location. Make a list of nans if the key is not present
    for sci_name in sci_names:
        if sci_name in location.keys():
            ret_list.append(location[sci_name])
        else:
            ret_list.append([np.nan for i in range(0, max_data_length)])
    return ret_list

# Return the nanmin, and nanmax of a list
# Checks to see if the data is all nans, if so it returns a nan for the min and max
# Primarily used to avoid taking the min/max of all NaN lists, which is common
def get_data_min_max(data):
    non_nans = np.count_nonzero(~np.isnan(data))
    if non_nans == 0:
        return [np.nan, np.nan]
    return [np.nanmin(data), np.nanmax(data)]

# Location of where the emails are downloaded
email_location = f'{Path(__file__).resolve().parents[1]}/littleproby_emails'

# Location of sbds
sbds_path = Path(f'{email_location}/sbds')

# Location to save dive netCDFs
netCDF_path = Path(f'{Path(__file__).resolve().parents[1]}/littleproby_emails/netCDFs')

# List of decoded jsons of the floats, located in the id folder of the sbds
decoded_jsons = []
for id_num in os.listdir(sbds_path):
    decoded_jsons.append(Path(f'{sbds_path}/{id_num}/{id_num}_decoded.json'))

# Open each JSON as a dict
for floaty, decoded_json in enumerate(decoded_jsons):
    with open(decoded_json) as f:
        decoded_dict = json.load(f)

    # If the JSON is non-empty
    if decoded_dict:

        dns = [] # List of valid dive numbers
        gps_entries = defaultdict(list) # Key=dive number, value is list of gps values (first is DiveStart, second is DiveEnd)
        dive_codes = defaultdict(list) # Key=dive number, value is list of code values (MPC, MPS, MNM, DTD)

        # Following dictionaries use the dive number (dn) as the key and
        # a list or dictionary as the value, correlated to the name of dictionary
        # Ex. {'1':[dive 1 data], '2':[dive 2 data], ...}
        parameters = defaultdict(list)
        units = defaultdict(dict)

        ascending_pressures = defaultdict(list)
        ascending_temperatures = defaultdict(list)
        ascending_salinities = defaultdict(list)
        ascending_tconds = defaultdict(list)

        parked_pressures = defaultdict(list)
        parked_temperatures = defaultdict(list)
        parked_salinities = defaultdict(list)

        surface_pressures = defaultdict(list)
        surface_temperatures = defaultdict(list)
        surface_salinities = defaultdict(list)
        surface_times = defaultdict(list)

        fall_time = defaultdict(list)
        fall_code = defaultdict(list)
        fall_pressure = defaultdict(list)

        rise_time = defaultdict(list)
        rise_code = defaultdict(list)
        rise_pressure = defaultdict(list)
    
        sn = decoded_dict[0]['sn']

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Data retrieval
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # For each valid dive (not -1 or 0), get the dive number, date, location and data.
        for dive in decoded_dict[0]['dives']:
            if dive['dn'] == -1: # Dive "-1" is an inital self-test
                last_self_test = dive['trajectory']['gps'][-1]['datetime']
                continue
            elif dive['dn'] < 1: # Dive "0" is a Diagnostic
                continue
            else:
                # Get dive number, start/end date, and start/end location
                for gps in dive['trajectory']['gps']:
                    # Continue if DiveStart or DiveEnd is not valid
                    # A dive is valid if it has lat/lon not None, its start date is after the self test date, and the dive has a start and end date
                    # Remove the dive number if it is present in the list of valid dive numbers
                    if ((gps['lat'] is None or gps['lon'] is None) and dive['dn'] not in dns) or (gps['datetime'] <= last_self_test):
                        if dive['dn'] in dns:
                            dns.remove(dive['dn'])
                        continue
                    elif gps['kind'] == 'DiveStart':
                        dns.append(dive['dn'])
                        gps_entries[dive['dn']].append(gps)
                    elif gps['kind'] == 'DiveEnd' and dive['dn'] in dns and gps not in gps_entries[dive['dn']]: # Only collect the end time if it's a valid dive (has a start time) and hasnt been included in gps_entries
                        gps_entries[dive['dn']].append(gps)
                
                # If theres only one gps_entry for a dive (no DiveStart or DiveEnd), remove it from the list of valid dives
                if len(gps_entries[dive['dn']]) == 1 and dive['dn'] in dns:
                    dns.remove(dive['dn'])

                # Get science data if current dive number has valid DiveStart/DiveEnd times and locations
                if dive['dn'] in dns:
                    units[dive['dn']] = dive['science']['units']
                    parameters[dive['dn']] = dive['status']['parameters']

                    # Get wanted dive parameters
                    MPC, MPS, MNM, DTD = ('','','','')
                    for p in dive['status']['parameters']:
                        if p['name'] == 'MPC':
                            MPC = p['value']
                        elif p['name'] == 'MPS':
                            MPS = p['value']
                        elif p['name'] == f'MNM{MPS}':
                            MNM = p['value']
                        elif p['name'] == f'DTD{MPS}':
                            DTD = p['value']
                    dive_codes[dive['dn']] = [MPC, MPS, MNM, DTD]

                    # Get Ascending data if present, otherwise set as NaN
                    if 'ascending' in dive['science'].keys():
                        data_lists = get_data(dive['science']['ascending']['binned'], ['pressure', 'temperature', 'salinity', 'tcond'])
                        ascending_pressures[dive['dn']] = data_lists[0]
                        ascending_temperatures[dive['dn']] = data_lists[1]
                        ascending_salinities[dive['dn']] = data_lists[2]
                        ascending_tconds[dive['dn']] = data_lists[3]
                    else:
                        ascending_pressures[dive['dn']] = [np.nan]
                        ascending_temperatures[dive['dn']] = [np.nan]
                        ascending_salinities[dive['dn']] = [np.nan]
                        ascending_tconds[dive['dn']] = [np.nan]

                    # Get Parked data if present, otherwise set as NaN
                    if 'park' in dive['science'].keys():
                        data_lists = get_data(dive['science']['park']['discrete'], ['pressure', 'temperature', 'salinity'])
                        parked_pressures[dive['dn']] = data_lists[0]
                        parked_temperatures[dive['dn']] = data_lists[1]
                        parked_salinities[dive['dn']] = data_lists[2]
                    else:
                        parked_pressures[dive['dn']] = [np.nan]
                        parked_temperatures[dive['dn']] = [np.nan]
                        parked_salinities[dive['dn']] = [np.nan]

                    # Get Surface data if present, otherwise set as NaN
                    if 'surface' in dive['science'].keys():
                        data_lists = get_data(dive['science']['surface']['discrete'], ['pressure', 'temperature', 'salinity', 'time'])
                        surface_pressures[dive['dn']] = data_lists[0]
                        surface_temperatures[dive['dn']] = data_lists[1]
                        surface_salinities[dive['dn']] = data_lists[2]
                        surface_times[dive['dn']] = data_lists[3]
                    else:
                        surface_pressures[dive['dn']] = [np.nan]
                        surface_temperatures[dive['dn']] = [np.nan]
                        surface_salinities[dive['dn']] = [np.nan]
                        surface_times[dive['dn']] = [np.nan]

                    # Get fallrise data
                    for i, code in enumerate(dive['trajectory']['fallrise']['code']):
                        if 'Descent' in code:
                            fall_time[dive['dn']].append(dive['trajectory']['fallrise']['time'][i])
                            fall_code[dive['dn']].append(code)
                            fall_pressure[dive['dn']].append(dive['trajectory']['fallrise']['pressure'][i])
                        elif 'Ascent' in code:
                            rise_time[dive['dn']].append(dive['trajectory']['fallrise']['time'][i])
                            rise_code[dive['dn']].append(code)
                            rise_pressure[dive['dn']].append(dive['trajectory']['fallrise']['pressure'][i])

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # netCDF creation
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        for dn in dns:
            # Profile coordinate: 4 digit probe id followed by 4 digit dive number
            # Ex: (19040003) -> Probe 1904, dive number 3
            profile_coord = {'profile':[int(f'{str(sn).zfill(4)}{str(dn).zfill(4)}')]}

            # Ascending science data arrays
            asc_coords = profile_coord | {'pressure_ascent':(('profile', 'asc_index'), [ascending_pressures[dn]])}
            asc_dims = ['profile', 'asc_index']

            T_asc_min_max = get_data_min_max(ascending_temperatures[dn])
            T_asc_attrs = {'long_name':'ascending sea water temperature', 'standard_name':'', 'units':'degrees_C', 'coverage_content_type':'physicalMeasurement', 'valid_min':T_asc_min_max[0], 'valid_max':T_asc_min_max[1]}
            T_asc = np.expand_dims(ascending_temperatures[dn], 0)
            T_asc = make_DA(T_asc, asc_dims, asc_coords, T_asc_attrs, ascending_pressures[dn], (sn, dn))
            T_asc.name = 'T_asc'

            S_asc_min_max = get_data_min_max(ascending_salinities[dn])
            S_asc_attrs = {'long_name':'ascending sea water salinity', 'standard_name':'', 'units':'1', 'coverage_content_type':'physicalMeasurement', 'valid_min':S_asc_min_max[0], 'valid_max':S_asc_min_max[1]}
            S_asc = np.expand_dims(ascending_salinities[dn], 0)
            S_asc = make_DA(S_asc, asc_dims, asc_coords, S_asc_attrs, ascending_pressures[dn], (sn, dn))
            S_asc.name = 'S_asc'

            TC_asc_min_max = get_data_min_max(ascending_tconds[dn])
            TC_asc_attrs = {'long_name':'ascending sea water tcond', 'standard_name':'', 'units':'degrees_C', 'coverage_content_type':'', 'valid_min':TC_asc_min_max[0], 'valid_max':TC_asc_min_max[1]}
            TC_asc = np.expand_dims(ascending_tconds[dn], 0)
            TC_asc = make_DA(TC_asc, asc_dims, asc_coords, TC_asc_attrs, ascending_pressures[dn], (sn, dn))
            TC_asc.name = 'TC_asc'

            ASC = xr.merge([T_asc, S_asc, TC_asc])

            # Parked science data arrays
            park_coords = profile_coord | {'pressure_park':(('profile', 'park_index'), [parked_pressures[dn]])}
            park_dims = ['profile', 'park_index']

            T_park_min_max = get_data_min_max(parked_temperatures[dn])
            T_park_attrs = {'long_name':'parked sea water temperature', 'standard_name':'', 'units':'degrees_C', 'coverage_content_type':'physicalMeasurement', 'valid_min':T_park_min_max[0], 'valid_max':T_park_min_max[1]}
            T_park = np.expand_dims(parked_temperatures[dn], 0)
            T_park = make_DA(T_park, park_dims, park_coords, T_park_attrs, parked_pressures[dn], (sn, dn))
            T_park.name = 'T_park'

            S_park_min_max = get_data_min_max(parked_salinities[dn])
            S_park_attrs = {'long_name':'parked sea water salinity', 'standard_name':'', 'units':'1', 'coverage_content_type':'physicalMeasurement', 'valid_min':S_park_min_max[0], 'valid_max':S_park_min_max[1]}
            S_park = np.expand_dims(parked_salinities[dn], 0)
            S_park = make_DA(S_park, park_dims, park_coords, S_park_attrs, parked_pressures[dn], (sn, dn))
            S_park.name = 'S_park'

            PARK = xr.merge([T_park, S_park])

            # Surface science data arrays
            surf_coords = profile_coord | {'pressure_surf':(('profile', 'surf_index'), [surface_pressures[dn]])}
            surf_dims = ['profile', 'surf_index']

            T_surf_min_max = get_data_min_max(surface_temperatures[dn])
            T_surf_attrs = {'long_name':'surface sea water temperature', 'standard_name':'', 'units':'degrees_C', 'coverage_content_type':'physicalMeasurement', 'valid_min':T_surf_min_max[0], 'valid_max':T_surf_min_max[1]}
            T_surf = np.expand_dims(surface_temperatures[dn], 0)
            T_surf = make_DA(T_surf, surf_dims, surf_coords, T_surf_attrs, surface_pressures[dn], (sn, dn))
            T_surf.name = 'T_surf'

            S_surf_min_max = get_data_min_max(surface_salinities[dn])
            S_surf_attrs = {'long_name':'surface sea water salinity', 'standard_name':'', 'units':'1', 'coverage_content_type':'physicalMeasurement', 'valid_min':S_surf_min_max[0], 'valid_max':S_surf_min_max[1]}
            S_surf = np.expand_dims(surface_salinities[dn], 0)
            S_surf = make_DA(S_surf, surf_dims, surf_coords, S_surf_attrs, surface_pressures[dn], (sn, dn))
            S_surf.name = 'S_surf'

            SURF = xr.merge([T_surf, S_surf])

            # Code arrays
            ref_time = np.datetime64('1970-01-01')
            time_rise_dt64 = sec_to_dt64(ref_time, rise_time[dn])
            time_fall_dt64 = sec_to_dt64(ref_time, fall_time[dn])

            Code_fall_coords = profile_coord | {'Code_fall_time': (('profile', 'fall_time'), [time_fall_dt64]), 'Code_fall_pressure':(('profile', 'fall_time'), [fall_pressure[dn]])}
            Code_fall_attrs = {'long_name':'code fall', 'standard_name':'', 'units':'', 'coverage_content_type':''}
            Code_fall = np.expand_dims(fall_code[dn], 0)
            Code_fall = make_DA(Code_fall, ['profile', 'fall_time'], Code_fall_coords, Code_fall_attrs, fall_pressure[dn], (sn, dn))
            Code_fall.name = 'Code_fall'

            Code_rise_coords = profile_coord | {'Code_rise_time': (('profile', 'rise_time'), [time_rise_dt64]), 'Code_rise_pressure':(('profile', 'rise_time'), [rise_pressure[dn]])}
            Code_rise_attrs = {'long_name':'code rise', 'standard_name':'', 'units':'', 'coverage_content_type':''}
            Code_rise = np.expand_dims(rise_code[dn], 0)
            Code_rise = make_DA(Code_rise, ['profile', 'rise_time'], Code_rise_coords, Code_rise_attrs, rise_pressure[dn], (sn, dn))
            Code_rise.name = 'Code_rise'

            CODE = xr.merge([Code_rise, Code_fall])

            # Merge components
            ds = xr.merge([PARK, SURF, ASC, CODE])

            # Add start/end lat/lon variables
            lat_attrs = {'units':'degrees_north', 'coverage_content_type':'coordinate', 'axis':'Y', 'valid_min':-90.0, 'valid_max':90.0}
            lon_attrs = {'units':'degrees_east', 'coverage_content_type':'coordinate', 'axis':'X', 'valid_min':-180.0, 'valid_max':180.0}
            ds['lat_start'] = xr.DataArray(data=[gps_entries[dn][0]['lat']], coords=profile_coord, dims=['profile'], attrs={'long_name':'latitude start', 'standard_name':''} | lat_attrs)
            ds['lat_end'] = xr.DataArray(data=[gps_entries[dn][1]['lat']], coords=profile_coord, dims=['profile'], attrs={'long_name':'latitude end', 'standard_name':''} | lat_attrs)
            ds['lon_start'] = xr.DataArray(data=[gps_entries[dn][0]['lon']], coords=profile_coord, dims=['profile'], attrs={'long_name':'latitude start', 'standard_name':''} | lon_attrs)
            ds['lon_end'] = xr.DataArray(data=[gps_entries[dn][1]['lon']], coords=profile_coord, dims=['profile'], attrs={'long_name':'latitude start', 'standard_name':''} | lon_attrs)

            # Coordinate attributes
            ds['profile'].attrs = {'cf_role':'profile_id'}
            asc_pressure_min_max = get_data_min_max(ascending_pressures[dn])
            ds['pressure_ascent'].attrs = {'long_name':'ascending pressure', 'standard_name':'', 'units':'dBar', 'coverage_content_type':'physicalMeasurement', 'valid_min':asc_pressure_min_max[0], 'valid_max':asc_pressure_min_max[1]}
            park_pressure_min_max = get_data_min_max(parked_pressures[dn])
            ds['pressure_park'].attrs = {'long_name':'parked pressure', 'standard_name':'', 'units':'dBar', 'coverage_content_type':'physicalMeasurement', 'valid_min':park_pressure_min_max[0], 'valid_max':park_pressure_min_max[1]}
            surf_pressure_min_max = get_data_min_max(surface_pressures[dn])
            ds['pressure_surf'].attrs = {'long_name':'surface pressure', 'standard_name':'', 'units':'dBar', 'coverage_content_type':'physicalMeasurement', 'valid_min':surf_pressure_min_max[0], 'valid_max':surf_pressure_min_max[1]}
            ds['Code_fall_time'].attrs = ds['Code_fall_time'].attrs | {'long_name':'fall time', 'standard_name':'', 'coverage_content_type':'coordinate', 'axis':'T'}
            fall_pressure_min_max = get_data_min_max(fall_pressure[dn])
            ds['Code_fall_pressure'].attrs = {'long_name':'code fall pressure', 'standard_name':'', 'units':'dBar', 'coverage_content_type':'physicalMeasurement', 'valid_min':fall_pressure_min_max[0], 'valid_max':fall_pressure_min_max[1]}
            ds['Code_rise_time'].attrs = ds['Code_rise_time'].attrs | {'long_name':'rise time', 'standard_name':'', 'coverage_content_type':'coordinate', 'axis':'T'}
            rise_pressure_min_max = get_data_min_max(rise_pressure[dn])
            ds['Code_rise_pressure'].attrs = {'long_name':'code fall pressure', 'standard_name':'', 'units':'dBar', 'coverage_content_type':'physicalMeasurement', 'valid_min':rise_pressure_min_max[0], 'valid_max':rise_pressure_min_max[1]}

            # Get min/max of all pressure readings through the whole dive
            all_pressures_min_max = get_data_min_max([] + asc_pressure_min_max + park_pressure_min_max + surf_pressure_min_max + fall_pressure_min_max + rise_pressure_min_max)

            # Dataset Metadata
            ds_attrs = {}
            ds_attrs['probe_id'] = sn
            ds_attrs['dive_number'] = dn
            ds_attrs['title'] = 'OMG Ocean ALAMO CTD Level 2 Data'
            ds_attrs['summary'] = 'This file contains salinity, temperature, and pressure measurements from air-deployed autonomous probes. Each product is divided into 3 regimes: ascending, parked, and surface, each with their own measurements and data.'
            ds_attrs['keywords'] = 'Water Temperature, Salinity'
            ds_attrs['keywords_vocabulary'] = 'NASA Global Change Master Directory (GCMD) Science Keywords'
            ds_attrs['Conventions'] = ''
            ds_attrs['id'] = ''
            ds_attrs['uuid'] = ''
            ds_attrs['naming_authority'] = 'gov.nasa.jpl'
            ds_attrs['cdm_data_type'] = ''
            ds_attrs['featureType'] = ''
            ds_attrs['history'] = 'JSON file created from dive specific sbd files transformed into NetCDF format.'
            ds_attrs['source'] = 'Data collected from air-deployed CTD instruments'
            ds_attrs['platform'] = ''
            ds_attrs['instrument'] = ''
            ds_attrs['processing_level'] = 'L2'
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
            ds_attrs['geospatial_lat_min'] = np.nanmin([gps_entries[dn][0]['lat'], gps_entries[dn][1]['lat']])
            ds_attrs['geospatial_lat_max'] = np.nanmax([gps_entries[dn][0]['lat'], gps_entries[dn][1]['lat']])
            ds_attrs['geospatial_lat_units'] = 'degrees_north'
            ds_attrs['geospatial_lat_resolution'] = 1.0E-7
            ds_attrs['geospatial_lon_min'] = np.nanmin([gps_entries[dn][0]['lon'], gps_entries[dn][1]['lon']])
            ds_attrs['geospatial_lon_max'] = np.nanmax([gps_entries[dn][0]['lon'], gps_entries[dn][1]['lon']])
            ds_attrs['geospatial_lon_units'] = 'degrees_east'
            ds_attrs['geospatial_lon_resolution'] = 1.0E-7
            ds_attrs['geospatial_vertical_min'] = all_pressures_min_max[0]
            ds_attrs['geospatial_vertical_max'] = all_pressures_min_max[1]
            ds_attrs['geospatial_vertical_resolution'] = ''
            ds_attrs['geospatial_vertical_units'] = 'dBar'
            ds_attrs['geospatial_vertical_positive'] = 'down'
            ds_attrs['time_coverage_start'] = gps_entries[dn][0]['datetime']
            ds_attrs['time_coverage_end'] = gps_entries[dn][1]['datetime']
            ds_attrs['time_coverage_duration'] = str(np.datetime64(gps_entries[dn][1]['datetime'][:-1]) - np.datetime64(gps_entries[dn][0]['datetime'][:-1]))
            ds_attrs['date_created'] = str(np.datetime64('now'))
            ds_attrs['source_product_header'] = ''
            ds_attrs['gps_dive_start_lat'] = [gps_entries[dn][0]['lat']]
            ds_attrs['gps_dive_end_lat'] = [gps_entries[dn][1]['lat']]
            ds_attrs['gps_dive_start_lon'] = [gps_entries[dn][0]['lon']]
            ds_attrs['gps_dive_end_lon'] = [gps_entries[dn][1]['lon']]
            ds_attrs['gps_dive_start_ttf'] = [gps_entries[dn][0]['ttf']]
            ds_attrs['gps_dive_end_ttf'] = [gps_entries[dn][1]['ttf']]
            ds_attrs['gps_dive_start_nsat'] = [gps_entries[dn][0]['nsat']]
            ds_attrs['gps_dive_end_nsat'] = [gps_entries[dn][1]['nsat']]
            ds_attrs['gps_dive_start_hdop'] = [gps_entries[dn][0]['hdop']]
            ds_attrs['gps_dive_end_hdop'] = [gps_entries[dn][1]['hdop']]
            ds_attrs['gps_dive_start_snr_min'] = [gps_entries[dn][0]['snr_min']]
            ds_attrs['gps_dive_end_snr_min'] = [gps_entries[dn][1]['snr_min']]
            ds_attrs['gps_dive_start_snr_mean'] = [gps_entries[dn][0]['snr_mean']]
            ds_attrs['gps_dive_end_snr_mean'] = [gps_entries[dn][1]['snr_mean']]
            ds_attrs['gps_dive_start_snr_max'] = [gps_entries[dn][0]['snr_max']]
            ds_attrs['gps_dive_end_snr_max'] = [gps_entries[dn][1]['snr_max']]
            for p in parameters[dn]:
                ds_attrs[f'param_{p["name"]}'] = p['value']

            # Update attributes of dataset
            ds.attrs = ds_attrs

            # Create directory if not present
            ds_path = f'{netCDF_path}/{sn}'
            if not os.path.exists(ds_path):
                os.makedirs(ds_path)

            # loop through data attributes
            # check if int32 or int64, encode it as int16
            # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH DATA VAR
            dv_encoding = dict()
            for dv in ds.data_vars:
                # netCDF does not support fill value for variable-length strings, ignore code_rise and code_fall
                if dv not in ['Code_rise', 'Code_fall']:
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

            # Save dataset with the name "XXXX_dive_NNNN_dataset_L2.nc" where XXXX is the probe id and NNNN is the dive number
            ds.to_netcdf(path=f'{ds_path}/{sn}_dive_{str(dn).zfill(4)}_dataset_L2.nc', encoding=encoding)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Dive table creation
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # dn     dive_start     dive_end     dive_time     start_lat     start_lon     end_lat     end_lon     MPC     MPS      MNM      DTD
        probe_csv_path = f'{netCDF_path}/{sn}'
        if not os.path.exists(probe_csv_path):
            continue
        probe_csv = open(f'{probe_csv_path}/{sn}_data_table.csv', 'w', newline='')
        probe_csv_writer = csv.writer(probe_csv, delimiter=',',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        probe_csv_writer.writerow(['dn', 'dive_start', 'dive_end', 'dive_time', 'start_lat', 'start_lon', 'end_lat', 'end_lon', 'MPC', 'MPS', 'MNM', 'DTD'])
        for dn in dns:
            MPC, MPS, MNM, DTD = dive_codes[dn]
            dive_start = gps_entries[dn][0]['datetime']
            dive_end = gps_entries[dn][1]['datetime']
            dive_time = str(np.datetime64(dive_end[:-1]) - np.datetime64(dive_start[:-1]))[:-8]
            start_loc = [gps_entries[dn][0]['lat'], gps_entries[dn][0]['lon']]
            end_loc = [gps_entries[dn][1]['lat'], gps_entries[dn][1]['lon']]
            probe_csv_writer.writerow([dn, dive_start, dive_end, dive_time, start_loc[0], start_loc[1], end_loc[0], end_loc[1], MPC, MPS, MNM, DTD])

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# JSON Format
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# [JSON] -> [{}]
#   "sn" 
#   "model"
#   [dives]
#       "dn"
#       "sw_version"
#       {trajectory}
#           [gps] -> [{}]* (only for valid dives)
#               "kind", "datetime", "lat", "lon", "ttf", "nst", "hdop", "snr_min", "snr_mean", "snr_max"
#           {fallrise}* (only for valid dives)
#               [time], [code], [pressure]
#       {science}
#           {units}
#               "pressure", "temperature", "salinity", "tcond", "time", "odo", "odo_phase", "odo_temperature", "par", "accel", "mag"
#           {ascending}* (only for valid dives)
#               {binned}
#                   [pressure]
#                   [temperature]
#                   [salinity]
#                   [tcond]
#           {park}* (only for valid dives)
#               {discrete}
#                   [pressure]
#                   [temperature]
#                   [salinity]
#           {surface}* (only for valid dives)
#               {discrete}
#                   [pressure]
#                   [temperature]
#                   [salinity]
#                   [time]
#       {status}
#           [parameters] -> [{}]
#               "name", "value"
#           [command_echo]* (not on all dives)
#           [argo]
#               "id", "kind"
#               [fields] -> [{}]
#                   "name", "value", "unit"
#           {engineering}
#               [pump] -> [{}]
#                   "name", [value], "unit"
#               [valve] -> [{}]
#                   "name", [value], "unit"
#               [other] -> [{}]
#                   "id", "kind"
#                   [fields] -> [{}]
#                       "name", "value", "unit"