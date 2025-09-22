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
    """Create a route based on train stops using actual railway line coordinates"""
    if not stops or not actual_stations:
        return []
    
    # Find the main railway line (longest track that connects most stations)
    main_line = find_main_railway_line()
    
    if not main_line:
        return create_simple_station_route(stops)
    
    # Get station positions on the main line
    route_points = []
    
    for stop_code in stops:
        if stop_code in actual_stations:
            station = actual_stations[stop_code]
            station_coord = [station['lon'], station['lat']]
            
            # Find the closest point on the main line to this station
            closest_point = find_closest_point_on_line(station_coord, main_line)
            route_points.append(closest_point)
    
    if len(route_points) < 2:
        return route_points
    
    # Create a continuous route along the main line between stations
    detailed_route = []
    
    for i in range(len(route_points) - 1):
        start_point = route_points[i]
        end_point = route_points[i + 1]
        
        # Extract the portion of main line between these points
        segment = extract_line_segment(main_line, start_point, end_point)
        
        if i == 0:
            detailed_route.extend(segment)
        else:
            # Skip the first point to avoid duplication
            detailed_route.extend(segment[1:])
    
    return detailed_route

def find_main_railway_line():
    """Build the main railway line by connecting track segments"""
    if not actual_tracks:
        return []
    
    # Get all main railway tracks
    main_tracks = []
    for track_id, track in actual_tracks.items():
        if (track.get('track_type') == 'rail' and 
            track.get('service', 'main') in ['main', None] and
            len(track.get('coordinates', [])) > 1):
            main_tracks.append(track['coordinates'])
    
    if not main_tracks:
        return []
    
    # If we have stations, try to build a route connecting them
    if actual_stations and len(actual_stations) >= 2:
        return build_connected_railway_line(main_tracks)
    
    # Otherwise, return the longest single track
    longest_track = max(main_tracks, key=len, default=[])
    return longest_track

def build_connected_railway_line(track_segments):
    """Connect track segments to build a continuous railway line"""
    if not track_segments:
        return []
    
    # Start with the track segment closest to SBC
    if 'SBC' in actual_stations:
        sbc_coord = [actual_stations['SBC']['lon'], actual_stations['SBC']['lat']]
        
        # Find the track segment closest to SBC
        closest_segment = None
        min_distance = float('inf')
        
        for segment in track_segments:
            for coord in segment:
                dist = math.sqrt((coord[0] - sbc_coord[0])**2 + (coord[1] - sbc_coord[1])**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_segment = segment
        
        if closest_segment:
            # Start building the route from this segment
            connected_line = list(closest_segment)
            used_segments = {id(closest_segment)}
            
            # Try to connect more segments
            current_end = connected_line[-1]
            
            while len(used_segments) < len(track_segments):
                best_segment = None
                best_distance = float('inf')
                best_reverse = False
                
                for segment in track_segments:
                    if id(segment) in used_segments:
                        continue
                    
                    # Check distance to start of segment
                    start_dist = math.sqrt(
                        (segment[0][0] - current_end[0])**2 + 
                        (segment[0][1] - current_end[1])**2
                    )
                    
                    # Check distance to end of segment (might need to reverse)
                    end_dist = math.sqrt(
                        (segment[-1][0] - current_end[0])**2 + 
                        (segment[-1][1] - current_end[1])**2
                    )
                    
                    if start_dist < best_distance:
                        best_distance = start_dist
                        best_segment = segment
                        best_reverse = False
                    
                    if end_dist < best_distance:
                        best_distance = end_dist
                        best_segment = segment
                        best_reverse = True
                
                # If we can't connect more segments (gap too large), break
                if best_distance > 0.01:  # ~1km gap
                    break
                
                if best_segment:
                    segment_to_add = list(reversed(best_segment)) if best_reverse else list(best_segment)
                    # Skip first point to avoid duplication
                    connected_line.extend(segment_to_add[1:])
                    current_end = connected_line[-1]
                    used_segments.add(id(best_segment))
            
            return connected_line
    
    # Fallback: return the longest segment
    return max(track_segments, key=len, default=[])

def find_closest_point_on_line(point, line_coords):
    """Find the closest point on a line to a given point"""
    if not line_coords:
        return point
    
    min_distance = float('inf')
    closest_point = line_coords[0]
    
    for coord in line_coords:
        distance = math.sqrt((point[0] - coord[0])**2 + (point[1] - coord[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_point = coord
    
    return closest_point

def extract_line_segment(line_coords, start_point, end_point):
    """Extract a segment of the line between two points"""
    if not line_coords or len(line_coords) < 2:
        return [start_point, end_point]
    
    # Find indices of closest points to start and end
    start_idx = 0
    end_idx = len(line_coords) - 1
    
    min_start_dist = float('inf')
    min_end_dist = float('inf')
    
    for i, coord in enumerate(line_coords):
        start_dist = math.sqrt((start_point[0] - coord[0])**2 + (start_point[1] - coord[1])**2)
        end_dist = math.sqrt((end_point[0] - coord[0])**2 + (end_point[1] - coord[1])**2)
        
        if start_dist < min_start_dist:
            min_start_dist = start_dist
            start_idx = i
        
        if end_dist < min_end_dist:
            min_end_dist = end_dist
            end_idx = i
    
    # Ensure correct order
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    
    # Extract the segment
    segment = line_coords[start_idx:end_idx + 1]
    
    # Ensure we have enough points for smooth movement
    if len(segment) < 5:
        # Interpolate more points
        interpolated = [segment[0]]
        for i in range(len(segment) - 1):
            start = segment[i]
            end = segment[i + 1]
            
            # Add interpolated points
            for j in range(1, 4):
                t = j / 4.0
                interpolated.append([
                    start[0] + (end[0] - start[0]) * t,
                    start[1] + (end[1] - start[1]) * t
                ])
            interpolated.append(end)
        return interpolated
    
    return segment

def create_simple_station_route(stops):
    """Fallback method to create route using station coordinates"""
    route = []
    for stop_code in stops:
        if stop_code in actual_stations:
            station = actual_stations[stop_code]
            route.append([station['lon'], station['lat']])
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
    """Load train data from CSV with proper track assignment"""
    global trains_data
    csv_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sbc_mys_schedules.csv')
    
    # First pass: load all train data without track assignment
    temp_trains = {}
    
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
                    
                    train_id = row['train_id']
                    
                    # Add some random delay for realism
                    import random
                    random_delay = random.uniform(0, 3) if row['train_type'] == 'passenger' else random.uniform(0, 8)
                    
                    temp_trains[train_id] = {
                        'train_id': train_id,
                        'dep_time': int(row['dep_time']),
                        'arr_time': int(row['arr_time']),
                        'speed_kmh': int(row['speed_kmh']),
                        'stops': stops,
                        'train_type': row['train_type'],
                        'priority': row['priority'],
                        'delay': random_delay
                    }
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing train {row['train_id']}: {e}")
                    continue
        
        # Second pass: assign tracks based on scheduling
        trains_data = assign_tracks_optimally(temp_trains)
        
        logger.info(f"Loaded {len(trains_data)} trains with optimal track assignment")
    except Exception as e:
        logger.error(f"Error loading train data: {e}")

def assign_tracks_optimally(temp_trains):
    """Assign tracks to minimize conflicts"""
    final_trains = {}
    track_schedules = {1: [], 2: []}  # Track occupied time slots
    
    # Sort trains by departure time for sequential assignment
    sorted_trains = sorted(temp_trains.items(), key=lambda x: x[1]['dep_time'])
    
    for train_id, train_data in sorted_trains:
        dep_time = train_data['dep_time'] + train_data['delay']
        arr_time = train_data['arr_time'] + train_data['delay']
        train_type = train_data['train_type']
        
        # Check track availability
        track1_available = is_track_available(track_schedules[1], dep_time, arr_time)
        track2_available = is_track_available(track_schedules[2], dep_time, arr_time)
        
        # Assign track based on availability and preference
        if train_type == 'passenger':
            # Passenger trains prefer Track 1
            if track1_available:
                assigned_track = 1
            elif track2_available:
                assigned_track = 2
            else:
                # Add delay to avoid conflict
                assigned_track = 1
                delay_needed = calculate_delay_needed(track_schedules[1], dep_time, arr_time)
                train_data['delay'] += delay_needed
                dep_time += delay_needed
                arr_time += delay_needed
        else:
            # Freight trains prefer Track 2
            if track2_available:
                assigned_track = 2
            elif track1_available:
                assigned_track = 1
            else:
                # Add delay to avoid conflict
                assigned_track = 2
                delay_needed = calculate_delay_needed(track_schedules[2], dep_time, arr_time)
                train_data['delay'] += delay_needed
                dep_time += delay_needed
                arr_time += delay_needed
        
        # Update track schedule
        track_schedules[assigned_track].append((dep_time, arr_time, train_id))
        
        # Finalize train data
        train_data['assigned_track'] = assigned_track
        train_data['track_name'] = f"Track {assigned_track}"
        final_trains[train_id] = train_data
    
    return final_trains

def is_track_available(track_schedule, dep_time, arr_time):
    """Check if track is available for the given time slot"""
    buffer = 3  # 3-minute buffer between trains
    
    for scheduled_dep, scheduled_arr, _ in track_schedule:
        # Check for overlap with buffer
        if not (arr_time + buffer <= scheduled_dep or dep_time >= scheduled_arr + buffer):
            return False
    
    return True

def calculate_delay_needed(track_schedule, dep_time, arr_time):
    """Calculate minimum delay needed to avoid conflicts"""
    if not track_schedule:
        return 0
    
    # Find the latest conflicting train
    latest_end = 0
    for scheduled_dep, scheduled_arr, _ in track_schedule:
        if scheduled_arr > dep_time and scheduled_dep < arr_time:
            latest_end = max(latest_end, scheduled_arr)
    
    if latest_end > 0:
        return latest_end - dep_time + 3  # 3-minute buffer
    
    return 0

def calculate_position(train_id, current_time):
    """Calculate train position ensuring it stays exactly on track coordinates"""
    if train_id not in trains_data:
        return None
    
    train = trains_data[train_id]
    
    # Get the actual railway track coordinates
    route_coords = create_train_route(train['stops'])
    
    if not route_coords or len(route_coords) < 2:
        logger.warning(f"No valid route found for train {train_id}")
        return None
    
    # Calculate timing based on realistic journey
    effective_dep_time = train['dep_time'] + train['delay']
    effective_arr_time = train['arr_time'] + train['delay']
    
    # Check if train hasn't started yet
    if current_time < effective_dep_time:
        return {
            'train_id': train_id,
            'lat': route_coords[0][1],  # Start at first track coordinate
            'lon': route_coords[0][0],
            'speed': 0,
            'status': 'waiting',
            'current_segment': train['stops'][0] if train['stops'] else 'Start',
            'next_station': train['stops'][1] if len(train['stops']) > 1 else train['stops'][0],
            'delay': train['delay'],
            'track_type': train.get('track_name', 'Track 1'),
            'assigned_track': train.get('assigned_track', 1),
            'train_type': train['train_type'],
            'progress': 0.0,
            'dwelling_time_remaining': 0
        }
    
    # Check if journey is completed
    if current_time > effective_arr_time:
        return {
            'train_id': train_id,
            'lat': route_coords[-1][1],  # End at last track coordinate
            'lon': route_coords[-1][0],
            'speed': 0,
            'status': 'completed',
            'current_segment': train['stops'][-1] if train['stops'] else 'End',
            'next_station': train['stops'][-1] if train['stops'] else 'End',
            'delay': train['delay'],
            'track_type': train.get('track_name', 'Track 1'),
            'assigned_track': train.get('assigned_track', 1),
            'train_type': train['train_type'],
            'progress': 1.0,
            'dwelling_time_remaining': 0
        }
    
    # Calculate progress along the route
    journey_time = effective_arr_time - effective_dep_time
    elapsed_time = current_time - effective_dep_time
    progress = min(1.0, elapsed_time / journey_time)
    
    # Find exact position on track coordinates
    track_position = calculate_track_position(route_coords, progress)
    
    # Calculate current speed
    base_speed = train['speed_kmh']
    current_speed = base_speed
    
    # Check if near any station (reduce speed)
    station_nearby = False
    current_station = None
    for station_code in train['stops']:
        if station_code in actual_stations:
            station = actual_stations[station_code]
            distance = math.sqrt(
                (track_position[0] - station['lon'])**2 + 
                (track_position[1] - station['lat'])**2
            )
            if distance < 0.005:  # Within ~500m of station
                station_nearby = True
                current_station = station_code
                current_speed = base_speed * 0.3  # Slow down near stations
                break
    
    status = 'dwelling' if station_nearby else 'running'
    
    # Determine next station
    next_station = train['stops'][-1]
    for i, station_code in enumerate(train['stops']):
        if station_code in actual_stations:
            station = actual_stations[station_code]
            distance = math.sqrt(
                (track_position[0] - station['lon'])**2 + 
                (track_position[1] - station['lat'])**2
            )
            if distance > 0.01:  # Not yet reached this station
                next_station = station_code
                break
    
    return {
        'train_id': train_id,
        'lat': track_position[1],  # Use exact track coordinate
        'lon': track_position[0],  # Use exact track coordinate
        'speed': current_speed if status == 'running' else 0,
        'status': status,
        'current_segment': current_station or f"{train['stops'][0]}-{train['stops'][-1]}",
        'next_station': next_station,
        'delay': train['delay'],
        'track_type': train.get('track_name', 'Track 1'),
        'assigned_track': train.get('assigned_track', 1),
        'train_type': train['train_type'],
        'progress': progress,
        'dwelling_time_remaining': 0
    }

def calculate_track_position(route_coords, progress):
    """Calculate exact position on track coordinates based on progress"""
    if not route_coords or len(route_coords) < 2:
        return route_coords[0] if route_coords else [0, 0]
    
    # Calculate position along the route
    total_points = len(route_coords)
    position_index = progress * (total_points - 1)
    
    # Get the two points to interpolate between
    lower_index = int(position_index)
    upper_index = min(lower_index + 1, total_points - 1)
    
    if lower_index == upper_index:
        return route_coords[lower_index]
    
    # Interpolate between the two track points
    local_progress = position_index - lower_index
    start_point = route_coords[lower_index]
    end_point = route_coords[upper_index]
    
    interpolated_lon = start_point[0] + (end_point[0] - start_point[0]) * local_progress
    interpolated_lat = start_point[1] + (end_point[1] - start_point[1]) * local_progress
    
    return [interpolated_lon, interpolated_lat]
    
    # Check if journey is completed
    if current_time > journey_plan[-1]['end_time']:
        return create_completed_position(train_id, train, route_coords[-1])
    
    # Default fallback
    return create_waiting_position(train_id, train, route_coords[0])

def create_realistic_journey_plan(train):
    """Create a realistic journey plan with proper timing for stations"""
    plan = []
    
    if len(train['stops']) < 2:
        return plan
    
    base_dep_time = train['dep_time'] + train['delay']
    current_time = base_dep_time
    
    # Calculate segment distances and times
    total_distance = 160  # SBC to MYS is approximately 160km
    
    for i in range(len(train['stops']) - 1):
        current_station = train['stops'][i]
        next_station = train['stops'][i + 1]
        
        # Calculate distance for this segment (simplified)
        segment_distance = total_distance / (len(train['stops']) - 1)
        
        # Calculate travel time based on speed
        travel_time = (segment_distance / train['speed_kmh']) * 60  # Convert to minutes
        
        # Add station waiting time (except for first station)
        if i > 0:
            waiting_time = get_station_waiting_time(current_station, train['train_type'], train['priority'])
            
            # Station stop segment
            plan.append({
                'type': 'station_stop',
                'station': current_station,
                'start_time': current_time,
                'end_time': current_time + waiting_time,
                'status': 'dwelling'
            })
            current_time += waiting_time
        
        # Travel segment
        plan.append({
            'type': 'travel',
            'from_station': current_station,
            'to_station': next_station,
            'start_time': current_time,
            'end_time': current_time + travel_time,
            'status': 'running'
        })
        current_time += travel_time
    
    # Final station stop
    final_station = train['stops'][-1]
    final_waiting = get_station_waiting_time(final_station, train['train_type'], train['priority'])
    plan.append({
        'type': 'station_stop',
        'station': final_station,
        'start_time': current_time,
        'end_time': current_time + final_waiting,
        'status': 'dwelling'
    })
    
    return plan

def calculate_position_in_segment(train_id, current_time, segment, route_coords):
    """Calculate exact position within a journey segment"""
    train = trains_data[train_id]
    
    if segment['type'] == 'station_stop':
        # Train is at station
        station_code = segment['station']
        
        # Generate events
        time_at_station = current_time - segment['start_time']
        if time_at_station < 0.5:  # Just arrived
            add_train_event(train_id, 'arriving', station_code, current_time)
        elif current_time > segment['end_time'] - 0.5:  # About to depart
            add_train_event(train_id, 'departing', station_code, current_time)
        
        if station_code in actual_stations:
            station = actual_stations[station_code]
            return {
                'train_id': train_id,
                'lat': station['lat'],
                'lon': station['lon'],
                'speed': 0,
                'status': 'dwelling',
                'current_segment': station_code,
                'next_station': station_code,
                'delay': train['delay'],
                'track_type': train.get('track_name', 'Track 1'),
                'assigned_track': train.get('assigned_track', 1),
                'train_type': train['train_type'],
                'progress': 0.0,
                'dwelling_time_remaining': segment['end_time'] - current_time
            }
    
    elif segment['type'] == 'travel':
        # Train is moving between stations
        segment_duration = segment['end_time'] - segment['start_time']
        elapsed_in_segment = current_time - segment['start_time']
        progress_in_segment = elapsed_in_segment / segment_duration
        
        # Find station coordinates for interpolation
        from_station = segment['from_station']
        to_station = segment['to_station']
        
        if from_station in actual_stations and to_station in actual_stations:
            from_coord = [actual_stations[from_station]['lon'], actual_stations[from_station]['lat']]
            to_coord = [actual_stations[to_station]['lon'], actual_stations[to_station]['lat']]
            
            # Find route segment between these stations
            route_section = find_route_between_stations(route_coords, from_coord, to_coord)
            
            if route_section and len(route_section) > 1:
                # Interpolate along the route section
                route_progress = progress_in_segment * (len(route_section) - 1)
                route_index = int(route_progress)
                local_progress = route_progress - route_index
                
                if route_index >= len(route_section) - 1:
                    lat, lon = route_section[-1][1], route_section[-1][0]
                else:
                    start_coord = route_section[route_index]
                    end_coord = route_section[route_index + 1]
                    lat = start_coord[1] + (end_coord[1] - start_coord[1]) * local_progress
                    lon = start_coord[0] + (end_coord[0] - start_coord[0]) * local_progress
            else:
                # Direct interpolation between stations
                lat = from_coord[1] + (to_coord[1] - from_coord[1]) * progress_in_segment
                lon = from_coord[0] + (to_coord[0] - from_coord[0]) * progress_in_segment
        else:
            # Fallback to route coordinates
            lat, lon = route_coords[0][1], route_coords[0][0]
        
        return {
            'train_id': train_id,
            'lat': lat,
            'lon': lon,
            'speed': train['speed_kmh'],
            'status': 'running',
            'current_segment': f"{from_station}-{to_station}",
            'next_station': to_station,
            'delay': train['delay'],
            'track_type': train.get('track_name', 'Track 1'),
            'assigned_track': train.get('assigned_track', 1),
            'train_type': train['train_type'],
            'progress': progress_in_segment,
            'dwelling_time_remaining': 0
        }
    
    # Fallback
    return create_waiting_position(train_id, train, route_coords[0])

def find_route_between_stations(route_coords, from_coord, to_coord):
    """Find the portion of route between two stations"""
    if not route_coords or len(route_coords) < 2:
        return [from_coord, to_coord]
    
    # Find closest points in route to stations
    from_index = 0
    to_index = len(route_coords) - 1
    
    min_from_dist = float('inf')
    min_to_dist = float('inf')
    
    for i, coord in enumerate(route_coords):
        from_dist = math.sqrt((coord[0] - from_coord[0])**2 + (coord[1] - from_coord[1])**2)
        to_dist = math.sqrt((coord[0] - to_coord[0])**2 + (coord[1] - to_coord[1])**2)
        
        if from_dist < min_from_dist:
            min_from_dist = from_dist
            from_index = i
        
        if to_dist < min_to_dist:
            min_to_dist = to_dist
            to_index = i
    
    # Ensure proper order
    if from_index > to_index:
        from_index, to_index = to_index, from_index
    
    # Return route section
    return route_coords[from_index:to_index + 1]

def create_waiting_position(train_id, train, coord):
    """Create position data for waiting train"""
    return {
        'train_id': train_id,
        'lat': coord[1],
        'lon': coord[0],
        'speed': 0,
        'status': 'waiting',
        'current_segment': train['stops'][0] if train['stops'] else 'SBC',
        'next_station': train['stops'][0] if train['stops'] else 'SBC',
        'delay': train['delay'],
        'track_type': train.get('track_name', 'Track 1'),
        'assigned_track': train.get('assigned_track', 1),
        'train_type': train['train_type'],
        'progress': 0.0,
        'dwelling_time_remaining': 0
    }

def create_completed_position(train_id, train, coord):
    """Create position data for completed train"""
    return {
        'train_id': train_id,
        'lat': coord[1],
        'lon': coord[0],
        'speed': 0,
        'status': 'completed',
        'current_segment': train['stops'][-1] if train['stops'] else 'MYS',
        'next_station': train['stops'][-1] if train['stops'] else 'MYS',
        'delay': train['delay'],
        'track_type': train.get('track_name', 'Track 1'),
        'assigned_track': train.get('assigned_track', 1),
        'train_type': train['train_type'],
        'progress': 1.0,
        'dwelling_time_remaining': 0
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
