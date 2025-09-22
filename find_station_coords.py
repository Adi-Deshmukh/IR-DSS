#!/usr/bin/env python3
"""
Script to find the closest track coordinates to each station
"""
import json
import math

def distance(coord1, coord2):
    """Calculate distance between two coordinates"""
    lat1, lon1 = coord1[1], coord1[0]
    lat2, lon2 = coord2[1], coord2[0]
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def find_closest_track_point(station_coord, tracks_data):
    """Find the closest track point to a station"""
    closest_point = None
    min_distance = float('inf')
    
    for feature in tracks_data['features']:
        if feature['geometry']['type'] == 'LineString':
            coordinates = feature['geometry']['coordinates']
            for coord in coordinates:
                dist = distance(station_coord, coord)
                if dist < min_distance:
                    min_distance = dist
                    closest_point = coord
    
    return closest_point, min_distance

# Load data
with open('bangalore_mysore_stations.geojson', 'r', encoding='utf-8') as f:
    stations_data = json.load(f)

with open('bangalore_mysore_tracks.geojson', 'r', encoding='utf-8') as f:
    tracks_data = json.load(f)

print("Finding closest track coordinates for each station:")
print("=" * 60)

updated_stations = []

for station in stations_data['features']:
    station_name = station['properties']['name']
    station_code = station['properties']['station_code']
    original_coord = station['geometry']['coordinates']
    
    closest_point, distance_km = find_closest_track_point(original_coord, tracks_data)
    
    print(f"\nStation: {station_name} ({station_code})")
    print(f"Original: [{original_coord[0]:.6f}, {original_coord[1]:.6f}]")
    print(f"Closest track: [{closest_point[0]:.6f}, {closest_point[1]:.6f}]")
    print(f"Distance: {distance_km*111:.1f} km")  # Convert degrees to approximate km
    
    # Update the station coordinate
    station['geometry']['coordinates'] = closest_point
    updated_stations.append(station)

# Create updated stations file
updated_data = {
    "type": "FeatureCollection",
    "features": updated_stations
}

with open('bangalore_mysore_stations_updated.geojson', 'w') as f:
    json.dump(updated_data, f, indent=2)

print(f"\nUpdated station coordinates saved to 'bangalore_mysore_stations_updated.geojson'")