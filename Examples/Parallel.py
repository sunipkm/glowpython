# %%
from time import perf_counter_ns
import pytz
from tqdm.contrib.concurrent import process_map, thread_map
from multiprocessing import Pool
from glowpython import no_precipitation, generic

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

res = list(map(generic, times, lats, lons, [100] * num_runs))
# res = thread_map(generic, (times, lats, lons, [100] * num_runs))
# %%
for ds in res:
    ds.ver.loc[dict(wavelength='5577')].plot()
    plt.show()