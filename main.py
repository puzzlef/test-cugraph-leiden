import os
import sys
import time
import rmm
import cudf
import cugraph


# Initialize RMM pool
pool = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=2**36)
rmm.mr.set_current_device_resource(pool)

# Read graph from file
file = os.path.expanduser(sys.argv[1])
print("Reading graph from file: {}".format(file), flush=True)
gdf  = cudf.read_csv(file, delimiter=' ', names=['src', 'dst'], dtype=['int32', 'int32'])
gdf  = cugraph.symmetrize_df(gdf, 'src', 'dst', None, False, False)
gdf["data"] = 1.0  # Add edge weights
G    = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='data', renumber=True)

# Run Leiden
parts, mod = cugraph.leiden(G)
for i in range(4):
  print("Running Leiden...", flush=True)
  t0 = time.time()
  parts, mod = cugraph.leiden(G)
  t1 = time.time()
  print("Leiden modularity: {:.6f}".format(mod), flush=True)
  print("Leiden took: {:.6f} s".format(t1-t0), flush=True)

# Save communities to file
comm = os.path.expanduser(sys.argv[2])
print("Saving communities to file: {}".format(comm), flush=True)
with open(comm, "w") as f:
  for i in range(len(parts)):
    f.write("{} {}\n".format(parts['vertex'].iloc[i], parts['partition'].iloc[i]))
