import os
import csv
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation

# Initialize points
def init():
    for point in start_points:
        point.set_data([], [])
    return start_points

# animation function.  This is called sequentially
def animate(i):
    if t_range:
        ax.set_title(f'Dives between {t_range[0]} and {t_range[1]}')
        ax.set_xlabel(f'Dive date: {t_start[i]}')
    else:
        ax.set_title(f'Dives between {min(t_start)} and {max(t_start)}')
        ax.set_xlabel(f'Dive date: {t_start[i]}')
    for j in range(0, i):
        x, y = m(start_lons[j], start_lats[j])
        start_points[j].set_data(x, y)
    return start_points

animated_plot = False
static_plot = True

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
fig.tight_layout(h_pad = 2)
m = Basemap(projection='lcc', width=2E6, height=3E6, lat_0=72, lon_0=-45) # Sets projection and viewing area
m.etopo(alpha=0.5)
marker_size = 3

# Location of dive netCDFs and CSVs
netCDF_path = Path(f'{Path(__file__).resolve().parents[1]}/littleproby_emails/netCDFs')

# List of decoded jsons of the floats, located in the id folder of the sbds
data_csvs = {}
id_nums = os.listdir(netCDF_path)
for id_num in id_nums:
    data_csvs[id_num] = Path(f'{netCDF_path}/{id_num}/{id_num}_data_table.csv')

# Time range for collecting and plotting dives, leave blank for all dates
# [start date, end date] (date format: yyyy-mm-ddThh:mm:ssZ)
# t_range = ['2017-01-01T00:00:00Z', '2018-01-01T00:00:00Z']
t_range = []

dns = defaultdict(list)
t_start = []
t_end = []
t_total = []
start_lats = []
start_lons = []
end_lats = []
end_lons = []
start_points = []
end_points = []
valid_id_nums = [] # id_nums with dives within time range
cmap = plt.get_cmap('viridis')
# colors = [cmap(i) for i in np.linspace(0, 1, len(id_nums))]
colors = ['darkred', 'blue', 'green', 'black']
markers = ['x', 'x', 'x', 'x']

id_colors = {}
id_markers = {}
for i, id_num in enumerate(id_nums):
    id_colors[id_num] = colors[i]
    id_markers[id_num] = markers[i]

start_plot_points = defaultdict(list)
end_plot_points = defaultdict(list)

for i, id_num in enumerate(id_nums):
    csv_path = data_csvs[id_num]

    probe_csv = open(csv_path, newline='')
    csv_reader = csv.reader(probe_csv, delimiter=' ', quotechar='|')
    for row in csv_reader:
        if 'dn' in row[0]:
            continue
        values = row[0].split(',')
        if not t_range or (values[1] >= t_range[0] and values[1] <= t_range[1]):
            if id_num not in valid_id_nums:
                valid_id_nums.append(id_num)
            dns[id_num].append(values[0])
            t_start.append(values[1])
            t_end.append(values[2])
            t_total.append(values[3])
            start_lats.append(values[4])
            start_lons.append(values[5])
            end_lats.append(values[6])
            end_lons.append(values[7])

            start_plot_points[id_num].append((values[5], values[4]))
            end_plot_points[id_num].append((values[7], values[6]))

    if animated_plot:
        x, y = m(0, 0)
        start_points.append(m.plot(x, y, 'x', markersize=marker_size, color=id_colors[id_num], label=f'{id_num} ({len(dns[id_num])} dives)')[0])
        for _ in dns[id_num][1:]:
            start_points.append(m.plot(x, y, 'x', color=id_colors[id_num], markersize=marker_size)[0])

if animated_plot:
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init,
                                frames=len(t_start), interval=150, blit=False, repeat_delay=3000)

# Makes static plot for dive starts within the given time range
# First dive point is an empty circle, subsequent dives are x's, and the final dive is a filled circle
if static_plot:
    for id_num in valid_id_nums:
        x, y = m(start_plot_points[id_num][0][0], start_plot_points[id_num][0][1])
        m.plot(x, y, 'o', mfc='none', markersize=marker_size, color=id_colors[id_num])
        x, y = m(start_plot_points[id_num][1][0], start_plot_points[id_num][1][1])
        m.plot(x, y, id_markers[id_num], markersize=marker_size, color=id_colors[id_num], label=f'{id_num} ({len(dns[id_num])} dives)')
        for (s_lon, s_lat) in start_plot_points[id_num][2:-1]:
            x, y = m(s_lon, s_lat)
            m.plot(x, y, id_markers[id_num], markersize=marker_size, color=id_colors[id_num])
        x, y = m(start_plot_points[id_num][-1][0], start_plot_points[id_num][-1][1])
        m.plot(x, y, 'o', markersize=marker_size, color=id_colors[id_num])
    if t_range:
        ax.set_title(f'Dives between {t_range[0]} and {t_range[1]}')
    else:
        ax.set_title(f'Dives between {min(t_start)} and {max(t_start)}')

plt.legend()
plt.show()
# anim.save('floats_animation.gif')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting (from parse_data.py, need to go through and remove/integrate it)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plotting = False

if plotting:
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 8))
    new_colors = [plt.get_cmap('jet')(i) for i in np.linspace(0, 1, len(dns))]
    custom_cycler = cycler('color', new_colors)

    # Ascending data sub-plot
    ax1.set_xlabel(f'Temperature ({units[1]["temperature"]})')
    ax1.set_ylabel(f'Pressure ({units[1]["pressure"]})')
    ax1.set_prop_cycle(custom_cycler)
    ax1.set_title('"Ascending"')
    ax1.invert_yaxis()
    ax1.grid()

    # Parked data sub-plot
    ax2.set_xlabel(f'Temperature ({units[1]["temperature"]})')
    ax2.set_prop_cycle(custom_cycler)
    ax2.set_title('"Parked"')
    ax2.invert_yaxis()
    ax2.grid()

    # Surface data sub-plot
    ax3.set_xlabel(f'Temperature ({units[1]["temperature"]})')
    ax3.set_prop_cycle(custom_cycler)
    ax3.set_title('"Surface"')
    ax3.invert_yaxis()
    ax3.grid()

    # Plot data dive-by-dive
    for dn in dns:
        if ascending_temperatures[dn] and ascending_pressures[dn]:
            ax1.plot(ascending_temperatures[dn], ascending_pressures[dn], linewidth=1)

        if parked_temperatures[dn] and parked_pressures[dn]:
            ax2.scatter(parked_temperatures[dn], parked_pressures[dn], s=.5)

        if surface_temperatures[dn] and surface_pressures[dn]:
            ax3.scatter(surface_temperatures[dn], surface_pressures[dn], s=.5)

    cax, _ = mpl.colorbar.make_axes(ax3, fraction=0.1)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('jet'), norm=mpl.colors.Normalize(vmin=min(dns), vmax=max(dns)))
    cb1.set_label('Dive number')
    plt.show()