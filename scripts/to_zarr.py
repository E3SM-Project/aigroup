import xarray as xr
import os
import time
import click
import xpartition


@click.command()
@click.option("--monthly-files", required=True, help="Path to directory containing monthly NetCDF files")
@click.option("--store-name", required=True, help="Name of the output Zarr store")
@click.option("--partitions", default=19, type=int, help="Number of partitions along time dimension")
def main(monthly_files, store_name, partitions):
    """Load NetCDF files from --monthly-files and write partitioned Zarr store to --store-name."""

    # Step 1: Load data
    start_time = time.time()
    print("Loading Data")
    ds = xr.open_mfdataset(os.path.join(monthly_files, "*.nc"))
    print("Data Loaded")
    print("Loading Time --- %s seconds ---" % (time.time() - start_time))

    # Step 2: Rechunk
    start_time = time.time()
    print("Rechuncking Starts")
    ds = ds.chunk(chunks={"time": partitions, "lat": 180, "lon": 360})
    print("Rechunking Done: Time --- %s seconds ---" % (time.time() - start_time))

    # Step 3: Write partitioned Zarr store
    partition_dims = ["time"]
    start_time = time.time()

    ds.partition.initialize_store(store_name)

    print("Partition Starts")
    for partition in range(partitions):
        print(f"Writing segment {partition + 1} / {partitions}")
        ds.partition.write(store_name, partitions, partition_dims, partition)
        print("Segment Time --- %s seconds ---" % (time.time() - start_time))

    print("Total Store Time --- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()


#python -u to_zarr.py \
#  --monthly-files monthly_files \
#  --store-name output.zarr \
#  --partitions 19
