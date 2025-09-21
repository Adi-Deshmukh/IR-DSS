"""
Improved Flask backend for Railway DSS
Uses actual GeoJSON files and provides better train movement visualization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import threading
import time
import csv
import math
import os
import ast
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
trains_data = {}
actual_stations = {}
actual_tracks = {}
simulation_running = False
simulation_time = 0
train_positions = {}
simulation_speed = 1
train_events = []  # Store train events for notifications

def get_station_waiting_time(station_code, train_type, train_priority):
    """Calculate waiting time at station based on train type and station characteristics"""
    base_times = {
        'SBC': {'passenger': 3, 'freight': 8},  # Main terminal - longer stops
        'MYA': {'passenger': 2, 'freight': 5},  # Intermediate major station
        'MYS': {'passenger': 3, 'freight': 8},  # Terminal station
        'Kengeri': {'passenger': 1, 'freight': 3},  # Suburban station
        'Bangarpet': {'passenger': 1, 'freight': 4},  # Junction
        'Channapatna': {'passenger': 1, 'freight': 3}  # Small station
    }
    
    base_time = base_times.get(station_code, {}).get(train_type, 2)
    
    # Adjust based on priority
    if train_priority == 'high':
        return base_time * 0.8  # High priority trains stop less
    elif train_priority == 'low':
        return base_time * 1.3  # Low priority trains wait longer
    else:
        return base_time

def add_train_event(train_id, event_type, station, timestamp):
    """Add a train event for notifications"""
    global train_events
    event = {
        'train_id': train_id,
        'event_type': event_type,  # 'arriving', 'departing', 'dwelling'
        'station': station,
        'timestamp': timestamp,
        'message': create_event_message(train_id, event_type, station)
    }
    train_events.append(event)
    
    # Keep only recent events (last 100)
    if len(train_events) > 100:
        train_events = train_events[-100:]

def create_event_message(train_id, event_type, station):
    """Create human-readable event message"""
    train = trains_data.get(train_id, {})
    train_type = train.get('train_type', 'train')
    track = train.get('track_name', 'Track 1')
    
    messages = {
        'arriving': f"🚂 {train_type.title()} Train {train_id} arriving at {station} on {track}",
        'departing': f"🚂 {train_type.title()} Train {train_id} departing from {station} on {track}",
        'dwelling': f"🚂 {train_type.title()} Train {train_id} dwelling at {station} on {track}"
    }
    
    return messages.get(event_type, f"Train {train_id} at {station}")

def load_actual_geojson():
    """Load actual GeoJSON files"""
    global actual_stations, actual_tracks
    
    # Load stations
    stations_file = os.path.join(os.path.dirname(__file__), '..', 'bangalore_mysore_stations.geojson')
    try:
        with open(stations_file, 'r', encoding='utf-8') as f:
            stations_data = json.load(f)
            for feature in stations_data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                actual_stations[props['station_code']] = {
                    'name': props['name'],
                    'code': props['station_code'],
                    'lat': coords[1],
                    'lon': coords[0],
                    'platforms': props.get('platforms', 2),
                    'major': props.get('platforms', 2) >= 5,
                    'dwell_time': props.get('dwell_time', 15)
                }
        logger.info(f"Loaded {len(actual_stations)} actual stations")
    except Exception as e:
        logger.error(f"Error loading stations: {e}")
    
    # Load tracks from GeoJSON
    tracks_file = os.path.join(os.path.dirname(__file__), '..', 'bangalore_mysore_tracks.geojson')
    try:
        with open(tracks_file, 'r', encoding='utf-8') as f:
            tracks_data = json.load(f)
            
        # Process all track features
        track_count = 0
        for i, feature in enumerate(tracks_data['features']):
            if feature['geometry']['type'] == 'LineString':
                coords = feature['geometry']['coordinates']
                if len(coords) >= 2:  # Valid track segment
                    track_id = f"track_{i}"
                    track_type = feature['properties'].get('railway', 'rail')
                    service = feature['properties'].get('service', 'main')
                    
                    actual_tracks[track_id] = {
                        'name': f'Track {i+1}',
                        'coordinates': coords,
                        'track_type': track_type,
                        'service': service,
                        'capacity': 1,
                        'segment': f'Segment-{i+1}',
                        'length_km': calculate_route_length(coords)
                    }
                    track_count += 1
        
        logger.info(f"Loaded {track_count} track segments from GeoJSON")
        
    except Exception as e:
        logger.error(f"Error loading tracks: {e}")
        # Fallback to simple route
        if actual_stations:
            simple_route = [
                [actual_stations['SBC']['lon'], actual_stations['SBC']['lat']],
                [actual_stations['MYA']['lon'], actual_stations['MYA']['lat']],
                [actual_stations['MYS']['lon'], actual_stations['MYS']['lat']]
            ]
            actual_tracks['main_route'] = {
                'name': 'SBC-MYS Simple Route',
                'coordinates': simple_route,
                'track_type': 'main',
                'capacity': 1,
                'segment': 'SBC-MYS',
                'length_km': calculate_route_length(simple_route)
            }

def create_interpolated_route(start_coords, end_coords, track_points):
    """Create interpolated route using actual track points"""
    route = [start_coords]
    
    # Add some intermediate points from actual tracks
    if track_points:
        # Sort track points by distance from start
        sorted_points = sorted(track_points, key=lambda p: 
            math.sqrt((p[0] - start_coords[0])**2 + (p[1] - start_coords[1])**2))
        
        # Add a few intermediate points
        for i in range(0, min(20, len(sorted_points)), 4):
            route.append(sorted_points[i])
    
    # Add intermediate stations
    if 'MYA' in actual_stations:
        route.append([actual_stations['MYA']['lon'], actual_stations['MYA']['lat']])
    
    route.append(end_coords)
    return route

def create_train_route(stops):
    """Create a route based on train stops using actual station coordinates and track data"""
    if not stops or not actual_stations:
        return []
    
    route = []
    
    # Add coordinates for each stop in order
    for stop_code in stops:
        if stop_code in actual_stations:
            station = actual_stations[stop_code]
            route.append([station['lon'], station['lat']])
        else:
            logger.warning(f"Station {stop_code} not found in actual stations")
    
    # If we have track data, create a detailed route following tracks
    if len(route) >= 2 and actual_tracks:
        enhanced_route = []
        
        for i in range(len(route) - 1):
            start_coord = route[i]
            end_coord = route[i + 1]
            
            # Add start coordinate
            enhanced_route.append(start_coord)
            
            # Find track segments that connect these stations
            connecting_points = find_connecting_track_points(start_coord, end_coord)
            
            # Add connecting track points
            if connecting_points:
                # Filter out points too close to start/end to avoid clustering
                filtered_points = []
                for point in connecting_points:
                    dist_to_start = math.sqrt((point[0] - start_coord[0])**2 + (point[1] - start_coord[1])**2)
                    dist_to_end = math.sqrt((point[0] - end_coord[0])**2 + (point[1] - end_coord[1])**2)
                    
                    # Only add points that are not too close to stations
                    if dist_to_start > 0.005 and dist_to_end > 0.005:
                        filtered_points.append(point)
                
                # Limit the number of points to prevent too much detail
                if len(filtered_points) > 20:
                    step = len(filtered_points) // 20
                    filtered_points = filtered_points[::step]
                
                enhanced_route.extend(filtered_points)
        
        # Add final coordinate
        enhanced_route.append(route[-1])
        
        # Smooth the route by removing points that create sharp turns
        smoothed_route = smooth_route(enhanced_route)
        return smoothed_route
    
    return route

def smooth_route(route_points):
    """Smooth the route by removing points that create unnecessary sharp turns"""
    if len(route_points) < 3:
        return route_points
    
    smoothed = [route_points[0]]  # Always keep first point
    
    for i in range(1, len(route_points) - 1):
        prev_point = smoothed[-1]
        current_point = route_points[i]
        next_point = route_points[i + 1]
        
        # Calculate distance from previous point
        distance_from_prev = math.sqrt(
            (current_point[0] - prev_point[0])**2 + 
            (current_point[1] - prev_point[1])**2
        )
        
        # Keep points that are far enough apart to prevent clustering
        if distance_from_prev > 0.003:  # Reduced threshold for smoother movement
            smoothed.append(current_point)
    
    smoothed.append(route_points[-1])  # Always keep last point
    
    # Ensure we have enough points for smooth movement
    if len(smoothed) < 10:
        # Add interpolated points for smoother movement
        final_route = [smoothed[0]]
        for i in range(len(smoothed) - 1):
            start = smoothed[i]
            end = smoothed[i + 1]
            
            # Add 3 interpolated points between each pair
            for j in range(1, 4):
                t = j / 4.0
                interpolated = [
                    start[0] + (end[0] - start[0]) * t,
                    start[1] + (end[1] - start[1]) * t
                ]
                final_route.append(interpolated)
            final_route.append(end)
        return final_route
    
    return smoothed

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points in degrees"""
    # Vector from p1 to p2
    v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    # Vector from p2 to p3  
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    
    # Calculate dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 180
    
    # Calculate angle in degrees
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

def find_connecting_track_points(start_coord, end_coord):
    """Find track points that lie between two stations using railway network"""
    connecting_points = []
    
    # Find the best track segments that connect the stations
    best_track_segments = find_best_track_route(start_coord, end_coord)
    
    # Collect all coordinates from the best track segments
    for segment in best_track_segments:
        connecting_points.extend(segment['coordinates'])
    
    if not connecting_points:
        # Fallback to previous bounding box method if no good route found
        min_lon = min(start_coord[0], end_coord[0]) - 0.01
        max_lon = max(start_coord[0], end_coord[0]) + 0.01
        min_lat = min(start_coord[1], end_coord[1]) - 0.01
        max_lat = max(start_coord[1], end_coord[1]) + 0.01
        
        for track in actual_tracks.values():
            if track['coordinates']:
                for coord in track['coordinates']:
                    lon, lat = coord[0], coord[1]
                    if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                        distance_to_line = point_to_line_distance(coord, start_coord, end_coord)
                        if distance_to_line < 0.02:
                            connecting_points.append(coord)
    
    # Remove duplicates and sort points to create a smooth path
    if connecting_points:
        # Remove duplicate points
        unique_points = []
        for point in connecting_points:
            if not unique_points or (abs(point[0] - unique_points[-1][0]) > 0.001 or 
                                   abs(point[1] - unique_points[-1][1]) > 0.001):
                unique_points.append(point)
        
        # Sort by distance from start to create a path
        unique_points.sort(key=lambda p: 
            math.sqrt((p[0] - start_coord[0])**2 + (p[1] - start_coord[1])**2))
        
        return unique_points
    
    return []

def find_best_track_route(start_coord, end_coord):
    """Find the best railway track segments that connect two stations"""
    # Define search radius around stations (in degrees, roughly 5km)
    search_radius = 0.05
    
    # Find tracks near start station
    start_tracks = []
    end_tracks = []
    
    for track_id, track in actual_tracks.items():
        if not track['coordinates'] or len(track['coordinates']) < 2:
            continue
            
        # Check if track has points near start station
        start_distance = min([
            math.sqrt((coord[0] - start_coord[0])**2 + (coord[1] - start_coord[1])**2)
            for coord in track['coordinates']
        ])
        
        # Check if track has points near end station  
        end_distance = min([
            math.sqrt((coord[0] - end_coord[0])**2 + (coord[1] - end_coord[1])**2)
            for coord in track['coordinates']
        ])
        
        if start_distance < search_radius:
            start_tracks.append((track_id, track, start_distance))
        if end_distance < search_radius:
            end_tracks.append((track_id, track, end_distance))
    
    # Sort by distance to stations
    start_tracks.sort(key=lambda x: x[2])
    end_tracks.sort(key=lambda x: x[2])
    
    # Find track segments that form a path between stations
    selected_segments = []
    
    # Look for tracks that connect both stations
    common_tracks = set([t[0] for t in start_tracks]) & set([t[0] for t in end_tracks])
    
    if common_tracks:
        # Use tracks that connect both stations
        for track_id in common_tracks:
            track = actual_tracks[track_id]
            if track['track_type'] == 'rail' and track.get('service', 'main') in ['main', 'siding']:
                selected_segments.append(track)
                break
    else:
        # Use a combination of nearby tracks
        # First add tracks near start station
        for track_id, track, dist in start_tracks[:3]:
            if track['track_type'] == 'rail' and track.get('service', 'main') in ['main', 'siding']:
                selected_segments.append(track)
        
        # Then add tracks near end station
        for track_id, track, dist in end_tracks[:3]:
            if track['track_type'] == 'rail' and track.get('service', 'main') in ['main', 'siding']:
                if track not in selected_segments:
                    selected_segments.append(track)
    
    return selected_segments

def point_to_line_distance(point, line_start, line_end):
    """Calculate distance from a point to a line segment"""
    px, py = point[0], point[1]
    x1, y1 = line_start[0], line_start[1]
    x2, y2 = line_end[0], line_end[1]
    
    # Calculate the distance
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return math.sqrt(A * A + B * B)
    
    param = dot / len_sq
    
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    dx = px - xx
    dy = py - yy
    return math.sqrt(dx * dx + dy * dy)

def calculate_route_length(coordinates):
    """Calculate route length in km"""
    total_length = 0
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i-1][1], coordinates[i-1][0]
        lat2, lon2 = coordinates[i][1], coordinates[i][0]
        
        # Haversine formula
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        total_length += R * c
    
    return total_length

def load_train_data():
    """Load train data from CSV"""
    global trains_data
    csv_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sbc_mys_schedules.csv')
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Safely parse the stops list
                    stops_str = row['stops'].strip()
                    if stops_str.startswith('[') and stops_str.endswith(']'):
                        stops = ast.literal_eval(stops_str)
                    else:
                        stops = [s.strip().strip("'") for s in stops_str.split(',')]
                    
                    # Assign track based on train ID and direction
                    train_id = row['train_id']
                    assigned_track = assign_track(train_id, stops, row['train_type'])
                    
                    # Add some random delay for realism (0-5 minutes)
                    import random
                    random_delay = random.uniform(0, 5) if row['train_type'] == 'passenger' else random.uniform(0, 10)
                    
                    trains_data[train_id] = {
                        'train_id': train_id,
                        'dep_time': int(row['dep_time']),
                        'arr_time': int(row['arr_time']),
                        'speed_kmh': int(row['speed_kmh']),
                        'stops': stops,
                        'train_type': row['train_type'],
                        'priority': row['priority'],
                        'delay': random_delay,  # Random initial delay
                        'assigned_track': assigned_track,  # Track assignment
                        'track_name': f"Track {assigned_track}"
                    }
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing train {row['train_id']}: {e}")
                    continue
        logger.info(f"Loaded {len(trains_data)} trains")
    except Exception as e:
        logger.error(f"Error loading train data: {e}")

def assign_track(train_id, stops, train_type):
    """Assign track based on train characteristics"""
    # Convert train_id to int for calculation
    train_num = int(train_id)
    
    # Different logic for different train types
    if train_type == 'passenger':
        # Passenger trains alternate between tracks based on ID
        if train_num % 2 == 0:
            return 1  # Even numbered trains on Track 1
        else:
            return 2  # Odd numbered trains on Track 2
    elif train_type == 'freight':
        # Freight trains mostly use Track 2 (goods line)
        return 2
    else:
        # Default assignment based on departure time
        # Early trains (even hours) on Track 1, odd hours on Track 2
        dep_minute = train_num % 120  # Use train number as proxy for time
        if dep_minute < 60:
            return 1
        else:
            return 2

def calculate_position(train_id, current_time):
    """Calculate train position with better movement visualization"""
    if train_id not in trains_data:
        return None
    
    train = trains_data[train_id]
    
    # Get track coordinates - create a comprehensive route
    if not actual_tracks or not actual_stations:
        return None
    
    # Create a route that connects all stations the train visits
    route_coords = create_train_route(train['stops'])
    
    if not route_coords or len(route_coords) < 2:
        # Fallback to first available track
        first_track = list(actual_tracks.values())[0]
        route_coords = first_track['coordinates']
    
    # Adjust for delays
    effective_dep_time = train['dep_time'] + train['delay']
    effective_arr_time = train['arr_time'] + train['delay']
    
    if current_time < effective_dep_time:
        # Not departed yet - show at starting station
        return {
            'train_id': train_id,
            'lat': route_coords[0][1],
            'lon': route_coords[0][0],
            'speed': 0,
            'status': 'waiting',
            'current_segment': train['stops'][0] if train['stops'] else 'SBC',
            'next_station': train['stops'][0] if train['stops'] else 'SBC',
            'delay': train['delay'],
            'track_type': train.get('track_name', 'Track 1'),
            'assigned_track': train.get('assigned_track', 1),
            'train_type': train['train_type'],
            'progress': 0.0
        }
    
    if current_time > effective_arr_time:
        # Journey completed
        return {
            'train_id': train_id,
            'lat': route_coords[-1][1],
            'lon': route_coords[-1][0],
            'speed': 0,
            'status': 'completed',
            'current_segment': train['stops'][-1] if train['stops'] else 'MYS',
            'next_station': train['stops'][-1] if train['stops'] else 'MYS',
            'delay': train['delay'],
            'track_type': train.get('track_name', 'Track 1'),
            'assigned_track': train.get('assigned_track', 1),
            'train_type': train['train_type'],
            'progress': 1.0
        }
    
    # Calculate progress along route with station stops
    journey_time = effective_arr_time - effective_dep_time
    elapsed_time = current_time - effective_dep_time
    base_progress = min(1.0, elapsed_time / journey_time)
    
    # Handle station stops - modify progress to account for waiting times
    actual_progress = base_progress
    status = 'running'
    current_station = None
    dwelling_time_remaining = 0
    
    # Check if we should be at a station
    if len(train['stops']) > 1:
        stops_timing = calculate_station_timing(train, journey_time)
        
        for i, (station_code, station_time, waiting_time) in enumerate(stops_timing):
            station_start_time = station_time
            station_end_time = station_time + waiting_time
            
            if station_start_time <= elapsed_time <= station_end_time:
                # Train is at this station
                status = 'dwelling'
                current_station = station_code
                dwelling_time_remaining = station_end_time - elapsed_time
                
                # Set position to station coordinates
                if station_code in actual_stations:
                    lat = actual_stations[station_code]['lat']
                    lon = actual_stations[station_code]['lon']
                
                # Generate arrival/departure events
                if not hasattr(train, 'last_event_station') or train.get('last_event_station') != station_code:
                    add_train_event(train_id, 'arriving', station_code, current_time)
                    train['last_event_station'] = station_code
                
                break
            elif elapsed_time > station_end_time and station_code == train.get('last_event_station'):
                # Train just departed
                add_train_event(train_id, 'departing', station_code, current_time)
                train['last_event_station'] = None
    
    # If not dwelling at station, interpolate position along route
    if status == 'running':
        # Interpolate position along route
        if len(route_coords) < 2:
            lat, lon = route_coords[0][1], route_coords[0][0]
        else:
            # Find which segment we're in
            segment_progress = actual_progress * (len(route_coords) - 1)
            segment_index = int(segment_progress)
            local_progress = segment_progress - segment_index
            
            if segment_index >= len(route_coords) - 1:
                lat, lon = route_coords[-1][1], route_coords[-1][0]
            else:
                # Linear interpolation
                start_coord = route_coords[segment_index]
                end_coord = route_coords[segment_index + 1]
                
                lat = start_coord[1] + (end_coord[1] - start_coord[1]) * local_progress
                lon = start_coord[0] + (end_coord[0] - start_coord[0]) * local_progress

def calculate_station_timing(train, total_journey_time):
    """Calculate when train arrives at each station and how long it waits"""
    if len(train['stops']) <= 1:
        return []
    
    # Distribute journey time across segments between stations
    total_segments = len(train['stops']) - 1
    segment_time = total_journey_time / total_segments
    
    station_timing = []
    current_time = 0
    
    for i, station_code in enumerate(train['stops']):
        if i == 0:
            # Starting station - no waiting time calculation needed
            continue
        elif i == len(train['stops']) - 1:
            # Final station - arrives but doesn't depart
            arrival_time = current_time + segment_time
            waiting_time = get_station_waiting_time(station_code, train['train_type'], train['priority'])
            station_timing.append((station_code, arrival_time, waiting_time))
        else:
            # Intermediate station
            arrival_time = current_time + segment_time
            waiting_time = get_station_waiting_time(station_code, train['train_type'], train['priority'])
            station_timing.append((station_code, arrival_time, waiting_time))
            current_time = arrival_time + waiting_time
    
    return station_timing

    # Determine next station and current segment
    next_station = 'MYS'
    current_segment = 'SBC-MYS'
    
    if current_station:
        next_station = current_station
        current_segment = current_station
    elif status == 'running':
        progress_through_stops = actual_progress * (len(train['stops']) - 1)
        stop_index = min(int(progress_through_stops) + 1, len(train['stops']) - 1)
        next_station = train['stops'][stop_index]
        
        # Set current segment
        if stop_index > 0:
            current_segment = f"{train['stops'][stop_index-1]}-{train['stops'][stop_index]}"
    
    # Calculate realistic speed based on status and track conditions
    current_speed = 0
    if status == 'running':
        # Vary speed based on track conditions and train type
        base_speed = train['speed_kmh']
        
        # Reduce speed near stations
        min_distance_to_station = min([
            math.sqrt((lat - station['lat'])**2 + (lon - station['lon'])**2)
            for station in actual_stations.values()
        ])
        
        if min_distance_to_station < 0.02:  # Within 2km of station
            current_speed = base_speed * 0.6  # Reduce to 60% speed
        else:
            current_speed = base_speed
    
    return {
        'train_id': train_id,
        'lat': lat,
        'lon': lon,
        'speed': current_speed,
        'status': status,
        'current_segment': current_segment,
        'next_station': next_station,
        'delay': train['delay'],
        'track_type': train.get('track_name', 'Track 1'),
        'assigned_track': train.get('assigned_track', 1),
        'train_type': train['train_type'],
        'progress': actual_progress,
        'dwelling_time_remaining': dwelling_time_remaining if status == 'dwelling' else 0
    }

def simulation_loop():
    """Enhanced simulation loop with configurable speed"""
    global simulation_running, simulation_time, train_positions
    
    while simulation_running:
        # Update train positions
        train_positions = {}
        for train_id in trains_data:
            pos = calculate_position(train_id, simulation_time)
            if pos:
                train_positions[train_id] = pos
        
        # Advance time based on simulation speed
        simulation_time += simulation_speed
        
        # Sleep time inversely proportional to simulation speed
        sleep_time = max(0.05, 0.5 / simulation_speed)  # Faster updates for higher speeds
        time.sleep(sleep_time)

# API Routes

@app.route('/')
def index():
    return jsonify({
        'name': 'Railway DSS - Improved Backend',
        'status': 'running' if simulation_running else 'stopped',
        'trains': len(trains_data),
        'stations': len(actual_stations),
        'tracks': len(actual_tracks),
        'time': simulation_time,
        'speed': simulation_speed
    })

@app.route('/start_sim', methods=['POST'])
def start_simulation():
    global simulation_running, simulation_time
    
    if not simulation_running:
        simulation_running = True
        if simulation_time == 0:  # Only reset time if starting fresh
            simulation_time = 0
        
        thread = threading.Thread(target=simulation_loop, daemon=True)
        thread.start()
        
        return jsonify({'success': True, 'message': 'Simulation started'})
    else:
        return jsonify({'success': True, 'message': 'Already running'})

@app.route('/stop_sim', methods=['POST'])
def stop_simulation():
    global simulation_running
    simulation_running = False
    return jsonify({'success': True, 'message': 'Simulation stopped'})

@app.route('/train_events')
def get_train_events():
    """Get recent train events for notifications"""
    global train_events
    
    # Return only events from the last 2 minutes
    current_time = simulation_time
    recent_events = [
        event for event in train_events 
        if current_time - event['timestamp'] <= 2.0
    ]
    
    return jsonify({
        'events': recent_events,
        'simulation_time': simulation_time
    })

@app.route('/reset_sim', methods=['POST'])
def reset_simulation():
    global simulation_running, simulation_time, train_positions, train_events
    simulation_running = False
    simulation_time = 0
    train_positions = {}
    train_events = []  # Clear events on reset
    
    # Reset train delays and event tracking
    for train in trains_data.values():
        train['delay'] = 0
        train.pop('last_event_station', None)  # Clear event tracking
    
    return jsonify({'success': True, 'message': 'Simulation reset'})

@app.route('/set_speed', methods=['POST'])
def set_simulation_speed():
    global simulation_speed
    data = request.get_json()
    new_speed = float(data.get('speed', 1.0))
    simulation_speed = max(0.1, min(10.0, new_speed))  # Limit between 0.1x and 10x
    return jsonify({'success': True, 'speed': simulation_speed})

@app.route('/positions', methods=['GET'])
def get_positions():
    return jsonify({
        'positions': train_positions,
        'timestamp': time.time(),
        'simulation_time': simulation_time,
        'speed': simulation_speed
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    active_trains = len([p for p in train_positions.values() if p['status'] == 'running'])
    completed_trains = len([p for p in train_positions.values() if p['status'] == 'completed'])
    delayed_trains = len([p for p in train_positions.values() if p['delay'] > 0])
    
    return jsonify({
        'stats': {
            'total_trains': len(trains_data),
            'active_trains': active_trains,
            'completed_trains': completed_trains,
            'on_time': completed_trains - delayed_trains,
            'delayed': delayed_trains,
            'avg_delay': sum(p['delay'] for p in train_positions.values()) / max(1, len(train_positions)),
            'throughput': active_trains,
            'simulation_time': simulation_time,
            'simulation_speed': simulation_speed
        }
    })

@app.route('/tracks', methods=['GET'])
def get_tracks():
    return jsonify({'tracks': actual_tracks})

@app.route('/stations', methods=['GET'])
def get_stations():
    return jsonify({'stations': actual_stations})

@app.route('/disrupt', methods=['POST'])
def add_disruption():
    data = request.get_json()
    train_id = data.get('train_id')
    delay_minutes = float(data.get('delay_minutes', 0))
    
    if train_id in trains_data:
        trains_data[train_id]['delay'] += delay_minutes
        return jsonify({
            'success': True,
            'message': f'Added {delay_minutes} min delay to {train_id}',
            'total_delay': trains_data[train_id]['delay']
        })
    else:
        return jsonify({'success': False, 'message': 'Train not found'})

@app.route('/special_train', methods=['POST'])
def add_special_train():
    data = request.get_json()
    train_id = data.get('train_id')
    
    if train_id not in trains_data:
        trains_data[train_id] = {
            'train_id': train_id,
            'dep_time': data.get('dep_time', 30),
            'arr_time': data.get('arr_time', 210),
            'speed_kmh': data.get('speed_kmh', 65),
            'stops': data.get('stops', ['SBC', 'MYA', 'MYS']),
            'train_type': 'special',
            'priority': 'high',
            'delay': 0
        }
        return jsonify({'success': True, 'message': f'Special train {train_id} added'})
    else:
        return jsonify({'success': False, 'message': 'Train ID already exists'})

if __name__ == '__main__':
    print("Starting Railway DSS - Improved Backend...")
    print("Loading GeoJSON data...")
    load_actual_geojson()
    
    print("Loading train schedules...")
    load_train_data()
    
    print(f"✓ Loaded {len(actual_stations)} stations")
    print(f"✓ Loaded {len(actual_tracks)} track segments") 
    print(f"✓ Loaded {len(trains_data)} trains")
    
    if not actual_stations or not actual_tracks or not trains_data:
        print("⚠️  Warning: Some data files may be missing or corrupted")
        print("   Make sure you have:")
        print("   - bangalore_mysore_stations.geojson")
        print("   - bangalore_mysore_tracks.geojson") 
        print("   - data/sbc_mys_schedules.csv")
    
    print("Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
