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
from typing import Dict, List, Tuple, Optional

# Import MILP optimizer and Live Controller
from milp_optimizer import RailwayMILPOptimizer, DecisionType, DecisionStatus, OptimizationDecision
from live_controller import LiveRailwayController

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Global State ---
trains_data: Dict[str, Dict] = {}
actual_stations: Dict[str, Dict] = {}
actual_tracks: Dict[str, Dict] = {}
simulation_running = False
simulation_time = 0
train_positions: Dict[str, Dict] = {}
simulation_speed = 1.0
train_events: List[Dict] = []
optimizer: Optional[RailwayMILPOptimizer] = None
live_controller: Optional[LiveRailwayController] = None
simulation_thread: Optional[threading.Thread] = None

# --- Helper Functions ---

def sanitize_for_json(obj):
    """Recursively remove problematic values for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, (DecisionType, DecisionStatus)):
        return obj.value
    return obj

def get_project_dir():
    """Get the absolute path to the project directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def add_train_event(train_id: str, event_type: str, details: str, timestamp: float):
    """Add a train event for notifications."""
    global train_events
    train = trains_data.get(train_id, {})
    message = f"Train {train_id} ({train.get('train_type', '')}): {details}"
    event = {
        'id': f"{train_id}-{event_type}-{timestamp}",
        'train_id': train_id,
        'event_type': event_type,
        'message': message,
        'timestamp': timestamp
    }
    train_events.append(event)
    # Keep only the last 100 events
    if len(train_events) > 100:
        train_events = train_events[-100:]

# --- Data Loading ---

def load_actual_geojson():
    """Load actual GeoJSON files for tracks and stations."""
    global actual_stations, actual_tracks
    project_dir = get_project_dir()
    
    # Load stations
    stations_file = os.path.join(project_dir, 'bangalore_mysore_stations.geojson')
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
                'major': props.get('platforms', 2) >= 5
            }
        logger.info(f"Loaded {len(actual_stations)} actual stations.")
    except Exception as e:
        logger.error(f"Error loading stations GeoJSON: {e}")

    # Load tracks
    tracks_file = os.path.join(project_dir, 'bangalore_mysore_tracks.geojson')
    try:
        with open(tracks_file, 'r', encoding='utf-8') as f:
            tracks_data = json.load(f)
        for i, feature in enumerate(tracks_data['features']):
            if feature['geometry']['type'] == 'LineString':
                track_id = f"track_{i}"
                actual_tracks[track_id] = {
                    'name': feature['properties'].get('name', f'Track {i+1}'),
                    'coordinates': feature['geometry']['coordinates'],
                    'track_type': feature['properties'].get('railway', 'rail'),
                    'service': feature['properties'].get('service', 'main'),
                    'length_km': calculate_route_length(feature['geometry']['coordinates'])
                }
        logger.info(f"Loaded {len(actual_tracks)} track segments.")
    except Exception as e:
        logger.error(f"Error loading tracks GeoJSON: {e}")

def load_train_data():
    """Load train data from CSV."""
    global trains_data
    project_dir = get_project_dir()
    csv_file = os.path.join(project_dir, 'data', 'sbc_mys_schedules.csv')
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    stops_str = row['stops'].strip()
                    if stops_str.startswith('[') and stops_str.endswith(']'):
                        stops = ast.literal_eval(stops_str)
                    else:
                        stops = [s.strip().strip("'") for s in stops_str.split(',')]
                    
                    train_id = row['train_id']
                    trains_data[train_id] = {
                        'train_id': train_id,
                        'dep_time': int(row['dep_time']),
                        'arr_time': int(row['arr_time']),
                        'speed_kmh': int(row['speed_kmh']),
                        'stops': stops,
                        'train_type': row['train_type'],
                        'priority': row['priority'],
                        'delay': 0.0, # Start with zero delay
                        'base_dep_time': int(row['dep_time']), # Store original times
                        'base_arr_time': int(row['arr_time']),
                    }
                except (ValueError, SyntaxError, KeyError) as e:
                    logger.error(f"Skipping malformed row in CSV for train {row.get('train_id', 'N/A')}: {e}")
        logger.info(f"Loaded {len(trains_data)} trains from CSV.")
    except Exception as e:
        logger.error(f"Fatal error loading train data: {e}")

# --- Simulation Logic ---

def calculate_route_length(coordinates: List[Tuple[float, float]]) -> float:
    """Calculate total length of a route in km using Haversine formula."""
    total_length = 0
    for i in range(len(coordinates) - 1):
        lon1, lat1 = coordinates[i]
        lon2, lat2 = coordinates[i+1]
        R = 6371  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        total_length += R * c
    return total_length

def find_main_railway_line() -> List[Tuple[float, float]]:
    """Find the longest continuous main track."""
    main_tracks = [t['coordinates'] for t in actual_tracks.values() if t.get('service') == 'main']
    return max(main_tracks, key=len) if main_tracks else []

MAIN_RAIL_LINE = []

def create_train_route(stops: List[str]) -> List[Tuple[float, float]]:
    """Create a route for a train along the main railway line."""
    if not stops or not actual_stations or not MAIN_RAIL_LINE:
        return []

    route_points = []
    for stop_code in stops:
        if stop_code in actual_stations:
            station = actual_stations[stop_code]
            station_coord = [station['lon'], station['lat']]
            
            # Find the point on the main line closest to this station
            closest_point_on_line = min(MAIN_RAIL_LINE, key=lambda p: math.hypot(p[0]-station_coord[0], p[1]-station_coord[1]))
            route_points.append(closest_point_on_line)
    
    return route_points

def calculate_position(train_id: str, current_time: float) -> Optional[Dict]:
    """Calculate train position based on its schedule and route."""
    if train_id not in trains_data:
        return None
    
    train = trains_data[train_id]
    route_coords = create_train_route(train['stops'])
    if not route_coords:
        return None
    
    effective_dep_time = train['dep_time'] + train['delay']
    effective_arr_time = train['arr_time'] + train['delay']

    if current_time < effective_dep_time:
        status = 'waiting'
        progress = 0.0
    elif current_time >= effective_arr_time:
        status = 'completed'
        progress = 1.0
    else:
        status = 'running'
        journey_duration = effective_arr_time - effective_dep_time
        time_elapsed = current_time - effective_dep_time
        progress = time_elapsed / journey_duration if journey_duration > 0 else 0

    # Determine position along the route geometry
    total_points = len(route_coords)
    idx_float = progress * (total_points - 1)
    idx1 = min(total_points - 2, int(idx_float))
    idx2 = idx1 + 1
    local_progress = idx_float - idx1

    lon1, lat1 = route_coords[idx1]
    lon2, lat2 = route_coords[idx2]
    
    lon = lon1 + (lon2 - lon1) * local_progress
    lat = lat1 + (lat2 - lat1) * local_progress

    # Determine current/next station
    current_segment = f"{train['stops'][0]}-{train['stops'][-1]}"
    next_station = train['stops'][-1]
    
    return {
        'train_id': train_id,
        'lat': lat,
        'lon': lon,
        'speed': train['speed_kmh'] if status == 'running' else 0,
        'status': status,
        'current_segment': current_segment,
        'next_station': next_station,
        'delay': train['delay'],
        'train_type': train['train_type'],
        'priority': train['priority'],
        'progress': progress
    }


def simulation_loop():
    """Main simulation loop to update train positions."""
    global simulation_running, simulation_time, train_positions
    
    while simulation_running:
        current_positions = {}
        for train_id in trains_data:
            pos = calculate_position(train_id, simulation_time)
            if pos:
                current_positions[train_id] = pos
        train_positions = current_positions

        if live_controller:
            live_controller.update_simulation_time(simulation_time)
            live_controller.update_train_positions(train_positions)

        simulation_time += simulation_speed
        
        sleep_duration = 0.5 / simulation_speed
        time.sleep(max(0.05, sleep_duration))

# --- API Endpoints ---

@app.route('/')
def index():
    return jsonify({
        'name': 'Railway DSS - Improved Backend',
        'status': 'running' if simulation_running else 'stopped',
        'trains': len(trains_data),
        'stations': len(actual_stations),
        'time': simulation_time,
    })

@app.route('/start_sim', methods=['POST'])
def start_simulation():
    global simulation_running, simulation_thread
    if not simulation_running:
        simulation_running = True
        simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
        simulation_thread.start()
        if live_controller:
            live_controller.start_live_control()
        return jsonify({'success': True, 'message': 'Simulation started.'})
    return jsonify({'success': True, 'message': 'Simulation already running.'})

@app.route('/stop_sim', methods=['POST'])
def stop_simulation():
    global simulation_running
    simulation_running = False
    if live_controller:
        live_controller.stop_live_control()
    return jsonify({'success': True, 'message': 'Simulation stopped.'})

@app.route('/reset_sim', methods=['POST'])
def reset_simulation():
    global simulation_running, simulation_time, train_positions, train_events, trains_data
    simulation_running = False
    time.sleep(0.2) # Allow loop to terminate
    simulation_time = 0
    train_positions = {}
    train_events = []
    # Reset train data to original state
    for train in trains_data.values():
        train['delay'] = 0.0
        train['dep_time'] = train['base_dep_time']
        train['arr_time'] = train['base_arr_time']
    if optimizer:
        optimizer.reset()
    logger.info("Simulation has been reset to its initial state.")
    return jsonify({'success': True, 'message': 'Simulation reset.'})


@app.route('/set_speed', methods=['POST'])
def set_simulation_speed():
    global simulation_speed
    data = request.get_json()
    new_speed = float(data.get('speed', 1.0))
    simulation_speed = max(0.1, min(10.0, new_speed))
    return jsonify({'success': True, 'speed': simulation_speed})

@app.route('/positions', methods=['GET'])
def get_positions():
    return jsonify(sanitize_for_json({
        'positions': train_positions,
        'timestamp': time.time(),
        'simulation_time': simulation_time
    }))

@app.route('/stats', methods=['GET'])
def get_stats():
    active_trains = len([p for p in train_positions.values() if p['status'] == 'running'])
    completed_trains = len([p for p in train_positions.values() if p['status'] == 'completed'])
    total_delay = sum(p['delay'] for p in train_positions.values())
    
    return jsonify(sanitize_for_json({
        'total_trains': len(trains_data),
        'active_trains': active_trains,
        'completed_trains': completed_trains,
        'avg_delay': total_delay / len(train_positions) if train_positions else 0
    }))

@app.route('/tracks', methods=['GET'])
def get_tracks():
    return jsonify({'tracks': actual_tracks})

@app.route('/stations', methods=['GET'])
def get_stations():
    return jsonify({'stations': actual_stations})

@app.route('/train_events', methods=['GET'])
def get_train_events():
    return jsonify({'events': train_events})

@app.route('/disrupt', methods=['POST'])
def add_disruption():
    data = request.get_json()
    train_id = data.get('train_id')
    delay_minutes = float(data.get('delay_minutes', 0))
    
    if not train_id or train_id not in trains_data:
        return jsonify({'success': False, 'message': 'Train not found'}), 404

    if optimizer:
        optimizer.apply_disruption(train_id, delay_minutes)
        add_train_event(train_id, 'disruption', f"Delayed by {delay_minutes} minutes", simulation_time)
        logger.info(f"Disruption: Applied {delay_minutes} min delay to train {train_id}.")
        return jsonify({
            'success': True, 
            'message': f'Added {delay_minutes} min delay to {train_id}. Re-optimization recommended.'
        })
    return jsonify({'success': False, 'message': 'Optimizer not available.'}), 500

@app.route('/optimize', methods=['POST'])
def optimize_schedule():
    if not optimizer:
        return jsonify({'success': False, 'message': 'Optimizer not available'}), 500

    data = request.get_json() or {}
    disrupted_trains = [t for t, d in trains_data.items() if d.get('delay', 0) > 0]

    logger.info("Running MILP optimization...")
    optimization_result = optimizer.optimize_schedule(
        disrupted_trains=disrupted_trains,
        strategy=int(data.get('strategy', 3))
    )

    if optimization_result.status == "optimal":
        # Decisions are now stored in the optimizer, to be fetched by /pending_decisions
        summary = optimizer.get_optimization_summary(optimization_result)
        logger.info(f"Optimization successful. Delay reduction: {summary['delay_reduction']:.2f} mins.")
        return jsonify(sanitize_for_json({
            'success': True,
            'message': 'Optimization successful. Review pending decisions for approval.',
            'summary': summary
        }))
    else:
        logger.warning(f"Optimization failed with status: {optimization_result.status}")
        return jsonify({
            'success': False,
            'message': f"Optimization failed: {optimization_result.status}"
        }), 500

@app.route('/pending_decisions', methods=['GET'])
def get_pending_decisions():
    if not optimizer:
        return jsonify({'decisions': [], 'message': 'Optimizer not available'}), 500
    
    pending = optimizer.get_pending_decisions()
    formatted_decisions = []
    for d in pending:
        impact = d.get('impact_analysis', {})
        delay_change = impact.get('delay_change', 0)
        impact_level = 'low'
        if abs(delay_change) > 10: impact_level = 'high'
        elif abs(delay_change) > 5: impact_level = 'medium'

        formatted_decisions.append({
            'id': d.get('decision_id'),
            'type': d.get('decision_type'),
            'train_id': d.get('train_id'),
            'station': d.get('station'),
            'description': f"Optimize schedule for Train {d.get('train_id')}",
            'impact': impact_level,
            'expected_savings': -delay_change if delay_change < 0 else 0,
            'details': d
        })
    
    return jsonify(sanitize_for_json({'success': True, 'decisions': formatted_decisions}))

@app.route('/approve_decision', methods=['POST'])
def approve_decision_endpoint():
    if not optimizer:
        return jsonify({'success': False, 'message': 'Optimizer not available'}), 500

    data = request.get_json()
    decision_id = data.get('decision_id')
    if not decision_id:
        return jsonify({'success': False, 'message': 'Decision ID is required'}), 400

    approved_decision = optimizer.approve_decision(decision_id)
    if not approved_decision:
        return jsonify({'success': False, 'message': 'Decision not found or already processed'}), 404

    # Apply the decision to the live simulation data
    if optimizer.apply_decision(approved_decision, trains_data):
        train_id = approved_decision.train_id
        logger.info(f"Decision {decision_id} for train {train_id} approved and applied.")
        add_train_event(train_id, 'decision_approved', f"Schedule optimized for train {train_id}", simulation_time)
        return jsonify({
            'success': True,
            'message': f'Decision {decision_id} approved and applied.'
        })
    else:
        return jsonify({'success': False, 'message': 'Failed to apply decision'}), 500

# --- Main Application Setup ---
if __name__ == '__main__':
    logger.info("Starting Railway DSS Backend...")
    
    logger.info("Loading GeoJSON data and train schedules...")
    load_actual_geojson()
    load_train_data()
    
    if not actual_tracks or not actual_stations or not trains_data:
        logger.critical("Failed to load essential data. Exiting.")
    else:
        MAIN_RAIL_LINE = find_main_railway_line()
        if not MAIN_RAIL_LINE:
            logger.warning("Could not determine main railway line from GeoJSON.")

        # Initialize the optimizer and controller
        optimizer = RailwayMILPOptimizer(trains_data, actual_stations, actual_tracks)
        live_controller = LiveRailwayController(trains_data, actual_stations, actual_tracks)
        
        logger.info(f"✓ Loaded {len(actual_stations)} stations, {len(actual_tracks)} track segments, {len(trains_data)} trains.")
        logger.info("Starting Flask server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)