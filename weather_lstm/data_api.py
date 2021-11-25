#
# Import Meteostat library and dependencies
#
from datetime import datetime
from meteostat import Point, Daily, Stations
import math


def get_weather_data_at_coordinates(latitude, longitude):

    start = datetime(datetime.today().year - 6, datetime.today().month, datetime.today().day)
    end = datetime.today()

    stations = Stations()
    stations = stations.nearby(latitude, longitude)
    stations_list = stations.fetch(50)
    print(stations_list)

    for i in range(0, 100):
        station = stations_list.iloc[[i]]
        data = Daily( station, start, end )
        data = data.fetch()
        if data.__len__() >= 365 * 6:
            data_array = data.to_numpy()
            complete = True
            for row in data_array:
                if math.isnan(row[2]):
                    complete = False
                    break
            if complete:
                break
    return data

#dataset = get_weather_data_at_coordinates(44.421, 26.044)
