"""
This script is used to identify 2D drought clusters for each time step separately. It parallelizes the for-loop through the
different time steps using mpi4py.

Here is an example of how this code should be run: 
mpirun -np 4 python 02_caculate_drought_clusters_parallel.py

Written by Julio E. Herrera Estrada, Ph.D.
"""

# Import libraries
import yaml
import numpy as np
from mpi4py import MPI
import pickle
from netCDF4 import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import pdb

# Import custom libraries
import drought_clusters_utils as dclib

# Initiate communicator and determine the core number and number of cores
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

##################################################################################
############################ SET PATHS AND DEFINITIONS ###########################
##################################################################################

# Load all the definitions needed to run this file
with open("definitions.yaml") as f:
    definitions = yaml.load(f, Loader=yaml.FullLoader)

# Name of the dataset used to calculate gridded drought metric
dataset = definitions["dataset"]

# Region where this analysis is carried out
region = definitions["region"]

# Name of the drought metric
drought_metric = definitions["drought_metric"]

# Threshold for drought definition
drought_threshold = float(definitions["drought_threshold"])
drought_threshold_name = str(drought_threshold)

# Start and end years for the timer period for which we will identify the drought clusters
start_year = definitions["start_year"]
end_year = definitions["end_year"]

# Set boolean variable of whether to treat the right/left edges of the map as periodic
periodic_bool = definitions["periodic_bool"]

# Path and file name of the NetCDF file with the drought metric
drought_metric_path = definitions["drought_metric_path"]
drought_metric_file_name = definitions["drought_metric_file_name"]

# Names of the variables in the NetCDF file with the drought metric
lat_var = definitions["lat_var"]
lon_var = definitions["lon_var"]
metric_var = definitions["metric_var"]

# Path where the drought clusters will be saved
clusters_partial_path = definitions["clusters_partial_path"]
clusters_full_path = os.path.join(
    clusters_partial_path,
    dataset,
    region,
    drought_metric,
    drought_threshold_name,
    ""
)

# Threshold for minimum cluster area (km^2)
minimum_area_threshold = definitions["minimum_area_threshold"]

# --- ENSURE OUTPUT DIRECTORY EXISTS ---
if rank == 0:
    os.makedirs(clusters_full_path, exist_ok = True)
comm.Barrier() # Wait for all cores to ensure the directory is created before proceeding

######################## DONE SETTING PATHS AND DEFINTIONS #######################

# Load the 3D array with the drought metric (t, lat, lon)
f = Dataset(os.path.join(drought_metric_path, drought_metric_file_name))
drought_metric = f.variables[metric_var][:]
lons = f.variables[lon_var][:]
lats = f.variables[lat_var][:]
f.close()

# # --- (DEBUG 1): INPUT DATA CHECK ---
# if rank == 0:
#     print(f"DEBUG 1.A (Data Input): Metric shape: {drought_metric.shape}")
#     print(f"DEBUG 1.B (Metrics): Data Max value: {np.nanmax(drought_metric)}", flush=True)
#     print(f"DEBUG 1.C (Metrics): Data Min value: {np.nanmin(drought_metric)}", flush=True)
#     print(f"DEBUG 1.D (Metrics): Count of NaN: {np.count_nonzero(np.isnan(drought_metric))}", flush=True)
# comm.Barrier()
# # ----------------------------------

# Set date time objects and the number of time steps
start_date = datetime(start_year, 1, 1)
nsteps = (end_year - start_year + 1) * 12
date_temp = start_date

# Spatial resolution of dataset in each direction
resolution_lon = np.abs(np.mean(lons[1:] - lons[:-1]))
resolution_lat = np.abs(np.mean(lats[1:] - lats[:-1]))

##################################################################################
#################### IDENTIFY DROUGHT CLUSTERS (PER TIME STEP) ###################
##################################################################################

# Function to carry out analysis in parallel. Each core is given a chunk of the time steps to analyze.
def find_clusters(chunk):

    # Length of the chunk
    chunk_length = len(chunk)
    
    # # --- (DEBUG 2): LOOP START CHECK ---
    # print(f"DEBUG 2: Rank {rank} received chunk of size {chunk_length} and is starting loop.")
    # # ----------------------------------
    
    # Repeat analysis for each time step within the assigned chunck
    for i in range(0, chunk_length):
        # Current date
        current_date = start_date + relativedelta(months=int(chunk[i]))
        date_str = current_date.strftime("%Y-%m-%d") 
        # STEP 1: GET DATA FOR THE CURRENT TIME STEP
        current_data_slice = drought_metric[int(chunk[i]), :, :]
        
        # STEP 2: APPLY MEDIAN FILTER TO THE TIME STEP IN EACH FIELD TO SMOOTH OUT NOISE
        # filtered_slice = dclib.median_filter(current_data_slice)
        filtered_slice = current_data_slice
        
        # STEP 3: APPLY DROUGHT THRESHOLD DEFINITION (e.g. 20th percentile)
        droughts = dclib.filter_non_droughts(filtered_slice, drought_threshold)

        # # --- (DEBUG 3): DROUGHT PIXEL COUNT ---
        # num_drought_pixels = np.count_nonzero(~np.isnan(droughts))
        # if rank == 0:
        #      print(f"DEBUG 3: Rank {rank}, DATE: {date_str} - DROUGHT PIXELS (Finite), {num_drought_pixels}", flush=True)
        # # --------------------------------------
        
        # STEP 4: IDENTIFY DROUGHT CLUSTERS PER TIME STEP
        # print(
        #     "Rank "
        #     + str(rank + 1)
        #     + ": Identifying clusters for time step "
        #     + str(int(chunk[i]) + 1)
        #     + " of "
        #     + str(nsteps)
        #     + " ("
        #     + str(i + 1)
        #     + "/"
        #     + str(chunk_length)
        #     + ")..."
        # )
        cluster_count_orig, cluster_dictionary = dclib.find_drought_clusters(
            droughts, lons, lats, resolution_lon, resolution_lat, periodic_bool
        )
        
        # # --- (DEBUG) 4.A: CLUSTER COUNT AFTER FIND BEFORE FILTER---
        # if rank == 0:
        #     print(f"DEBUG 4.A.Rank {rank}: --- DATA BEFORE FILTER: {date_str} ---", flush=True)
        #     print(f"DEBUG 4.A: Rank {rank}, Clusters Found (Initial Count), {cluster_count_orig}")
        #     print(f"DEBUG 4.A.Rank {rank}: 'droughts' (Input Data Mask) Shape: {droughts.shape}", flush=True)
        #     print(f"DEBUG 4.5.Rank {rank}: 'cluster_dictionary' (Cluster Properties):", flush=True)
            
        #     # Print cluster properties, focusing on the calculated area
        #     for k, v in cluster_dictionary.items():
        #         area = v.get('area', 0.0)
        #         lat, lon = v.get('centroid', ('N/A', 'N/A'))
        #         print(f"  DEBUG 4.5.Rank {rank}:   Cluster {k} (Area): {area} km^2, Centroid: ({lat}, {lon})", flush=True)
        #     print(f"  DEBUG 4.5.Rank {rank}: ---------------------------------------", flush=True)
        # # ------------------------------------------

        # STEP 5: FILTER DROUGHT CLUSTERS BY AREA AND IF THE CENTROID LIES IN THE SAHARA
        droughts, cluster_count, cluster_dictionary = dclib.filter_drought_clusters(
            droughts, cluster_count_orig, cluster_dictionary, minimum_area_threshold
        )

        # # --- (DEBUG) 4.B: CLUSTER COUNT AFTER FILTER ---
        # if rank == 0:
        #     print(f"DEBUG 4.B: Rank {rank}: Clusters Remaining (Post-Filter Count): {cluster_count}")
        # # ---------------------------------------------
        
        # STEP 6: SAVE THE DROUGHT CLUSTERS FOR CURRENT TIME STEP
        # Paths and file names for saving data
        f_name_slice = os.path.join(
            clusters_full_path , "cluster-matrix_"+ date_str + ".pck"
        )
        f_name_dictionary = os.path.join(
            clusters_full_path , "cluster-dictionary_" + date_str + ".pck"
        )
        f_name_count = os.path.join(
            clusters_full_path , "cluster-count_" + date_str + ".pck"
        )

        # # --- (DEBUG) 5: CHECK THRESHOLD CRITERIA ---
        # print(f"DEBUG 5.Rank {rank}: Slice Max (Raw): {np.nanmax(current_data_slice)}")
        # print(f"DEBUG 5.Rank {rank}: Slice Min (Raw): {np.nanmin(current_data_slice)}")
        # print(f"DEBUG 5.Rank {rank}: Threshold: {drought_threshold}")
        
        # # Check the output of the filtering step, 'droughts', which is about to be saved
        # num_drought_pixels = np.count_nonzero(~np.isnan(droughts))
        # if rank == 0:
        #     print(f"DEBUG 5.Rank {rank}: DATE: {date_str} - DROUGHT PIXELS (Finite): {num_drought_pixels}", flush=True)
        # # -------------------------------

        # # --- (DEBUG) 6: PRE-SAVE CHECK ---
        # print(f"DEBUG 6: Rank {rank} is attempting to save file: {f_name_slice}")
        # # -------------------------------
                
        # Save the data in pickle format
        pickle.dump(droughts, open(f_name_slice, "wb"), pickle.HIGHEST_PROTOCOL)
        pickle.dump(
            cluster_dictionary, open(f_name_dictionary, "wb"), pickle.HIGHEST_PROTOCOL
        )
        pickle.dump(cluster_count, open(f_name_count, "wb"), pickle.HIGHEST_PROTOCOL)
        # print(
        #     "Rank "
        #     + str(rank + 1)
        #     + ": Saved data for time step "
        #     + str(int(chunk[i]) + 1)
        #     + " of "
        #     + str(nsteps)
        #     + " ("
        #     + str(i + 1)
        #     + "/"
        #     + str(chunk_length)
        #     + ")."
        # )

    return


# Number of steps for each processor
offset = 0
h = np.ceil(nsteps / np.float32(size - offset))

# Number of steps that each process will be required to do
if rank >= offset and rank < size - 1:
    chunk = np.arange((rank - offset) * h, (rank - offset) * h + h)
elif rank == size - 1:
    chunk = np.arange((rank - offset) * h, nsteps)

# Identify drought clusters for the current chunk of data
find_clusters(chunk)
