import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

birddata = pd.read_csv("bird_tracking.csv")

#Plot the paths of these 3 birds
bird_names = pd.unique(birddata.bird_name)
plt.figure(figsize=(7,7))
for bird in bird_names:
    ix = birddata.bird_name == bird
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    plt.plot(x,y,".", label=bird)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="lower right")
plt.savefig("3traj.pdf")


#Plotting the speed of Eric the bird
plt.figure(figsize=(8,4))
speed = birddata.speed_2d[birddata.bird_name == "Eric"]
#the list `speed` contains some NaN type objects (85 to be exact)
ind = np.isnan(speed) 
#Plot the histogram with all the indices NOT contained in `ind`
#ind has all the NaN object by asking np.isnan() 
plt.hist(speed[~ind], bins=np.linspace(0,30,20), normed = True) 
plt.xlabel("2D speed (m/s)")
plt.ylabel("Frequency");
plt.savefig("hist.pdf")


#Plotting using pandas
birddata.speed_2d.plot(kind="hist", range=[0,30])
plt.xlabel("2D speed")
plt.savefig("pd_hist.pdf")


#Plotting the mean daily speed of Eric the bird
date_str = birddata.date_time[0]
datetime.datetime.strptime(date_str[:-3], "%Y-%m-%d %H:%M:%S") #convert simple str into a datetime object
timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime(birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))

birddata["timestamp"] = pd.Series(timestamps, index = birddata.index)

data = birddata[birddata.bird_name == 'Eric']
times = data.timestamp
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)


next_day = 1
inds = []
daily_mean_speed = []
for (i,t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = []


plt.figure(figsize=(8,6))
plt.plot(daily_mean_speed)
plt.xlabel("day")
plt.ylabel("Mean speed (m/s)");
plt.savefig("Daily mean.pdf")

#after this it uses Cartopy to restore the image of bird migration as
#it has stretched when converting into a 2d image.
#cannot download Cartopy, see screenshot for code
#also compared migration track against a map for clearance


        
        
