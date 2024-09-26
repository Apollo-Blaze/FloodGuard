from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
city = Point(38.3498, -81.6326, 200)  # Latitude, Longitude, Elevation in meters

# Set the date for which you want to retrieve the data
start = datetime(2023, 6, 24)
end = datetime(2023, 6, 24)

# Get daily data for 2018
data = Daily(city, start, end)
data = data.fetch()
print(data)

