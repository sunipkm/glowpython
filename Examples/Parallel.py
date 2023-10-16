# %%
from time import perf_counter_ns
import numpy as np
import pytz
from tqdm.contrib.concurrent import process_map, thread_map
from multiprocessing import Pool
from glowpython import no_precipitation, generic
from functools import partial
import sys

from datetime import datetime, timedelta
from matplotlib import pyplot as plt
# %%
num_runs = 100
num_pars = num_runs // 4

times = [datetime(2022, 1, 28, 0, 0, 0, tzinfo=pytz.utc) + timedelta(seconds=s) for s in range(0, 86400, 86400 // 4)] * num_pars
lats = [0] * num_runs
lons = [0] * num_runs
f107 = [70] * num_runs
ap = [4] * num_runs
meta = [dict(metadata=i) for i in range(num_runs)]
params = [(time, lat, lon, 100, dict(metadata=i)) for time, lat, lon, i in zip(times, lats, lons, range(num_runs))]
# %%
def wrap_generic(params):
    return generic(params[0], params[1], params[2], params[3], **params[4])

# res = list(map(generic, times, lats, lons, [100] * num_runs))
start = perf_counter_ns()
res = list(map(wrap_generic, params))
end = perf_counter_ns()
print(f'Iterations: {num_runs}, Time taken: {(end - start)*1e-9:.6f} s')
sys.exit(0)
# res = thread_map(wrap_generic, params)
# %%
mnum = []
for ds in res:
    mnum.append(ds.attrs['metadata'])
mnum = np.array(mnum)
print('Sorted:', np.all(np.diff(mnum) >= 0))
# %%
for ds in res:
    fig, ax = plt.subplots()
    fig.suptitle('%s: %s'%(ds.attrs['metadata'], ds.attrs['time']))
    ds.ver.loc[dict(wavelength='5577')].plot(ax=ax)
    plt.show()
# %%
