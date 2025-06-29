# sidewalk-guardian
Smart detection of trash accumulation to trigger removal request

## Install Dependencies
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
or 
```
pip3 install pandas geopandas folium scikit-learn mapillary shapely tqdm requests
```

## Source data:
[NYC OpenData](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/explore/query/SELECT%0A%20%20%60unique_key%60%2C%0A%20%20%60created_date%60%2C%0A%20%20%60closed_date%60%2C%0A%20%20%60agency%60%2C%0A%20%20%60agency_name%60%2C%0A%20%20%60complaint_type%60%2C%0A%20%20%60descriptor%60%2C%0A%20%20%60location_type%60%2C%0A%20%20%60incident_zip%60%2C%0A%20%20%60incident_address%60%2C%0A%20%20%60street_name%60%2C%0A%20%20%60cross_street_1%60%2C%0A%20%20%60cross_street_2%60%2C%0A%20%20%60intersection_street_1%60%2C%0A%20%20%60intersection_street_2%60%2C%0A%20%20%60address_type%60%2C%0A%20%20%60city%60%2C%0A%20%20%60landmark%60%2C%0A%20%20%60facility_type%60%2C%0A%20%20%60status%60%2C%0A%20%20%60due_date%60%2C%0A%20%20%60resolution_description%60%2C%0A%20%20%60resolution_action_updated_date%60%2C%0A%20%20%60community_board%60%2C%0A%20%20%60bbl%60%2C%0A%20%20%60borough%60%2C%0A%20%20%60x_coordinate_state_plane%60%2C%0A%20%20%60y_coordinate_state_plane%60%2C%0A%20%20%60open_data_channel_type%60%2C%0A%20%20%60park_facility_name%60%2C%0A%20%20%60park_borough%60%2C%0A%20%20%60vehicle_type%60%2C%0A%20%20%60taxi_company_borough%60%2C%0A%20%20%60taxi_pick_up_location%60%2C%0A%20%20%60bridge_highway_name%60%2C%0A%20%20%60bridge_highway_direction%60%2C%0A%20%20%60road_ramp%60%2C%0A%20%20%60bridge_highway_segment%60%2C%0A%20%20%60latitude%60%2C%0A%20%20%60longitude%60%2C%0A%20%20%60location%60%0AWHERE%0A%20%20caseless_eq%28%60complaint_type%60%2C%20%22Obstruction%22%29%0A%20%20AND%20caseless_contains%28%60descriptor%60%2C%20%22Trash%22%29%0A%20%20AND%20caseless_eq%28%60location_type%60%2C%20%22Sidewalk%22%29%0A%20%20AND%20%60created_date%60%0A%20%20%20%20%20%20%20%20BETWEEN%20%222024-06-21T15%3A34%3A15%22%20%3A%3A%20floating_timestamp%0A%20%20%20%20%20%20%20%20AND%20%222025-06-21T15%3A34%3A15%22%20%3A%3A%20floating_timestamp%0AORDER%20BY%20%60created_date%60%20DESC%20NULL%20FIRST%0ASEARCH%20%22trash%22/page/filter)

I used the online tool to make a similar query to this:
```
SELECT created_date, latitude, longitude
FROM "311"
WHERE complaint_type='Obstruction'
  AND descriptor ILIKE '%Trash%'
  AND descriptor ILIKE '%Sidewalk%'
  AND created_date > CURRENT_DATE - INTERVAL '1 year';

```

Create data visualization by running:
```
python3 visualize_reports.py
```


## Inference
Use Moondream query to infere number of trash bags in the picture. If number is greater than 3, generate a 311 request. 