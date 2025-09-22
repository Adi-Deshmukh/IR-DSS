"""
Enhanced MILP Optimizer for Railway Decision Support System
Based on Törnquist & Persson (2007) with improvements for real-world application
Fixed implementation with geopandas, rerouting, and through destinations
"""

import pulp
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
from typing import Dict, List, Tuple, Optional, Set, Union
import logging
import json
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import original optimizer components
try:
    from milp_optimizer import (
        TrainEvent, DecisionType, DecisionStatus, 
        OptimizationDecision, OptimizationResult
    )
    # Try to import RailwayTrack if it exists
    try:
        from milp_optimizer import RailwayTrack
    except ImportError:
        RailwayTrack = None
except ImportError as e:
    print(f"Failed to import from milp_optimizer: {e}")
    # Define minimal classes if import fails
    
    @dataclass
    class TrainEvent:
        event_id: str
        train_id: str
        segment_id: str
        is_station: bool
        planned_arrival: float
        planned_departure: float
        min_duration: float
        is_planned_stop: bool
        origin_direction: int = 0
        is_forced: bool = False
    
    class DecisionType(Enum):
        TIME_ADJUSTMENT = "time_adjustment"
        TRACK_ASSIGNMENT = "track_assignment"
        ORDER_CHANGE = "order_change"
        REROUTING = "rerouting"
    
    class DecisionStatus(Enum):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"
    
    @dataclass
    class OptimizationDecision:
        decision_id: str
        decision_type: str
        train_id: str
        event_id: str
        segment_id: str
        description: str
        impact_score: float
        original_value: any
        new_value: any
        confidence: float
        status: str
        created_at: datetime
    
    @dataclass
    class OptimizationResult:
        status: str
        objective_value: float
        total_delay: float
        delay_reduction: float
        decisions: list
        optimized_times: dict
        track_assignments: dict
        order_changes: list
        rerouting_decisions: list
        computation_time: float
        solver_status: str
    
    RailwayTrack = None

# Define RailwayTrack if not imported
if RailwayTrack is None:
    @dataclass
    class RailwayTrack:
        track_id: str
        segment_id: str
        length: float
        capacity: int
        track_lengths: list
        follow_separation: float = 5.0
        meet_separation: float = 10.0
import math
import time
import uuid
import ast
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    PLATFORM_ASSIGNMENT = "platform_assignment"
    DEPARTURE_OPTIMIZATION = "departure_optimization" 
    FULL_OPTIMIZATION = "full_optimization"
    EMERGENCY_REROUTING = "emergency_rerouting"
    TRACK_ASSIGNMENT = "track_assignment"
    ORDER_CHANGE = "order_change"
    REROUTING = "rerouting"

class DecisionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"

@dataclass
class OptimizationDecision:
    decision_id: str
    decision_type: DecisionType
    timestamp: float
    train_id: str
    station: Optional[str]
    original_value: Dict
    proposed_value: Dict
    priority: str
    impact_analysis: Dict
    status: DecisionStatus = DecisionStatus.PENDING
    approval_reason: Optional[str] = None
    
    def to_dict(self):
        result = asdict(self)
        result['decision_type'] = self.decision_type.value
        result['status'] = self.status.value
        return result

@dataclass
class TrainEvent:
    """Event k in the MILP model - represents a train visiting a segment"""
    event_id: str
    train_id: str
    segment_id: str
    is_station: bool
    planned_arrival: float  # e_initial_k (minutes from start)
    planned_departure: float  # b_initial_k (minutes from start)
    min_duration: float  # d_k (minimum time in minutes)
    is_planned_stop: bool  # h_k (1 if scheduled stop)
    origin_direction: int  # o_k (0 or 1 for direction)
    is_ongoing: bool = False  # For events already started
    is_forced: bool = False   # For fixed events that cannot change

@dataclass
class SegmentData:
    """Segment j in the MILP model - represents a track segment (line or station)"""
    segment_id: str
    name: str
    distance_km: float
    num_tracks: int  # |P_j| - number of tracks on this segment
    is_station: bool  # s_j (1 if station, 0 if line)
    track_lengths: List[float]  # g^track_jt for each track (meters)
    meet_separation: float  # DM_j - minimum separation for meeting trains (minutes)
    follow_separation: float  # DF_j - minimum separation for following trains (minutes)
    coordinates: List[Tuple[float, float]]  # GPS coordinates for distance calculation
    alternative_segments: List[str] = None  # Alternative routes for rerouting

@dataclass
class OptimizationResult:
    """Results of MILP optimization"""
    status: str
    objective_value: float
    total_delay: float
    delay_reduction: float
    decisions: List[OptimizationDecision]
    optimized_times: Dict[str, Dict]  # train_id -> {arrival_times, departure_times}
    track_assignments: Dict[str, int]  # event_id -> track_number
    order_changes: List[Dict]
    rerouting_decisions: List[Dict]
    computation_time: float
    solver_status: str

class EnhancedRailwayMILPOptimizer:
    """
    Enhanced MILP Optimizer implementing Törnquist & Persson (2007) model
    with geopandas integration, rerouting, and through destination support
    """
    
    def __init__(self, trains_data: Dict, stations: Dict, tracks: Dict):
        self.trains_data = trains_data
        self.stations = stations
        self.raw_tracks = tracks
        
        # Enhanced track processing with geopandas
        self.tracks = self._process_enhanced_track_data()
        
        # MILP Model Sets (from paper)
        self.T = set()  # Set of trains
        self.B = set()  # Set of segments  
        self.E = set()  # Set of events
        self.K = {}     # K_i: Events for train i (ordered)
        self.L = {}     # L_j: Events on segment j
        self.P = {}     # P_j: Tracks on segment j
        self.R = {}     # R_j: Alternative segments for rerouting
        
        # MILP Parameters
        self.M = 10000  # Big-M constant (increased for stability)
        self.w = {}     # w_i: Penalty threshold (minutes)
        self.c_low = {}    # c^low_i: Low delay cost per minute
        self.c_penalty = {}  # c^penalty_i: Penalty cost for delays > w_i
        self.c_reroute = {}  # c^reroute_i: Rerouting cost
        self.g_train = {}    # g^train_i: Train length (meters)
        self.g_connection = {}  # g^connection_kk': Connection time
        
        # Distance calculation with geopandas
        self.distance_matrix = {}
        self.segment_geometries = {}
        
        # Decision tracking
        self.pending_decisions = []
        self.decision_history = []
        self.decision_counter = 0
        
        # Optimization parameters
        self.max_optimization_time = 60  # seconds (increased)
        self.strategy = 3  # Strategy 3 from paper (limited swaps)
        
        # Initialize enhanced MILP model
        self._initialize_enhanced_model()
        
    def _process_enhanced_track_data(self) -> Dict[str, SegmentData]:
        """Process track data with geopandas for real distance calculation"""
        processed_tracks = {}
        
        try:
            # Load GeoJSON data if available
            station_coords = {}
            for station_code, station in self.stations.items():
                if 'lat' in station and 'lon' in station:
                    station_coords[station_code] = (station['lon'], station['lat'])
                elif 'coordinates' in station:
                    coords = station['coordinates']
                    station_coords[station_code] = (coords[0], coords[1])
            
            # Define main railway segments with real coordinates
            segments_config = [
                # Line segments
                ("SBC_Kengeri", "SBC-Kengeri Line", False, 2, 25.0),
                ("Kengeri_MYA", "Kengeri-MYA Line", False, 2, 60.0),
                ("MYA_Channapatna", "MYA-Channapatna Line", False, 1, 45.0),
                ("Channapatna_MYS", "Channapatna-MYS Line", False, 1, 55.0),
                
                # Alternative routes for rerouting
                ("SBC_MYA_ALT", "SBC-MYA Alternative", False, 1, 90.0),
                ("Kengeri_Channapatna_ALT", "Kengeri-Channapatna Bypass", False, 1, 75.0),
                
                # Station segments
                ("SBC", "SBC Station", True, 6, 0.0),
                ("Kengeri", "Kengeri Station", True, 3, 0.0),
                ("MYA", "MYA Station", True, 4, 0.0),
                ("Channapatna", "Channapatna Station", True, 2, 0.0),
                ("MYS", "MYS Station", True, 6, 0.0),
                
                # Through destination extensions
                ("MYS_Erode", "MYS-Erode Extension", False, 1, 120.0),
                ("MYS_Chennai", "MYS-Chennai Extension", False, 1, 180.0),
            ]
            
            for seg_id, name, is_station, num_tracks, distance in segments_config:
                # Calculate real coordinates if available
                coordinates = []
                if is_station and seg_id in station_coords:
                    coord = station_coords[seg_id]
                    coordinates = [coord, coord]  # Station is a point
                elif not is_station:
                    # For line segments, use start and end station coordinates
                    parts = seg_id.split('_')
                    if len(parts) >= 2:
                        start_station = parts[0]
                        end_station = parts[1]
                        if start_station in station_coords and end_station in station_coords:
                            coordinates = [station_coords[start_station], station_coords[end_station]]
                
                # Calculate real distance using geopandas if coordinates available
                real_distance = distance
                if len(coordinates) >= 2 and not is_station:
                    try:
                        real_distance = geodesic(
                            (coordinates[0][1], coordinates[0][0]),  # lat, lon format for geodesic
                            (coordinates[1][1], coordinates[1][0])
                        ).kilometers
                        logger.info(f"Real distance for {seg_id}: {real_distance:.1f} km")
                    except:
                        logger.warning(f"Could not calculate real distance for {seg_id}, using default: {distance} km")
                
                # Track lengths (500m for stations, actual length for lines)
                track_lengths = [500.0 if is_station else real_distance * 1000] * num_tracks
                
                # Separation times based on track type and importance
                if is_station:
                    meet_sep = 2.0 if num_tracks >= 4 else 3.0  # minutes
                    follow_sep = 1.5 if num_tracks >= 4 else 2.0  # minutes
                else:
                    meet_sep = 5.0 if num_tracks >= 2 else 8.0  # minutes
                    follow_sep = 3.0 if num_tracks >= 2 else 5.0  # minutes
                
                # Define alternative segments for rerouting
                alternative_segments = []
                if seg_id == "SBC_Kengeri":
                    alternative_segments = ["SBC_MYA_ALT"]
                elif seg_id == "Kengeri_MYA":
                    alternative_segments = ["Kengeri_Channapatna_ALT"]
                elif seg_id == "MYA_Channapatna":
                    alternative_segments = ["Kengeri_Channapatna_ALT"]
                
                processed_tracks[seg_id] = SegmentData(
                    segment_id=seg_id,
                    name=name,
                    distance_km=real_distance,
                    num_tracks=num_tracks,
                    is_station=is_station,
                    track_lengths=track_lengths,
                    meet_separation=meet_sep,
                    follow_separation=follow_sep,
                    coordinates=coordinates,
                    alternative_segments=alternative_segments
                )
                
                logger.info(f"Created segment {seg_id}: {num_tracks} tracks, {real_distance:.1f}km")
            
        except Exception as e:
            logger.error(f"Error processing enhanced track data: {e}")
            # Fallback to simple processing
            processed_tracks = self._process_simple_track_data()
            
        return processed_tracks
    
    def _process_simple_track_data(self) -> Dict[str, SegmentData]:
        """Fallback simple track processing"""
        logger.warning("Using fallback simple track processing")
        
        simple_segments = {
            "SBC": SegmentData("SBC", "SBC Station", 0.0, 6, True, [500]*6, 2.0, 1.5, [], []),
            "Kengeri": SegmentData("Kengeri", "Kengeri Station", 0.0, 3, True, [500]*3, 3.0, 2.0, [], []),
            "MYA": SegmentData("MYA", "MYA Station", 0.0, 4, True, [500]*4, 2.5, 1.8, [], []),
            "Channapatna": SegmentData("Channapatna", "Channapatna Station", 0.0, 2, True, [500]*2, 3.0, 2.0, [], []),
            "MYS": SegmentData("MYS", "MYS Station", 0.0, 6, True, [500]*6, 2.0, 1.5, [], []),
            "SBC_Kengeri": SegmentData("SBC_Kengeri", "SBC-Kengeri", 25.0, 2, False, [25000]*2, 5.0, 3.0, [], []),
            "Kengeri_MYA": SegmentData("Kengeri_MYA", "Kengeri-MYA", 60.0, 2, False, [60000]*2, 5.0, 3.0, [], []),
            "MYA_Channapatna": SegmentData("MYA_Channapatna", "MYA-Channapatna", 45.0, 1, False, [45000], 8.0, 5.0, [], []),
            "Channapatna_MYS": SegmentData("Channapatna_MYS", "Channapatna-MYS", 55.0, 1, False, [55000], 8.0, 5.0, [], []),
        }
        
        return simple_segments
    
    def _initialize_enhanced_model(self):
        """Initialize enhanced MILP sets and parameters"""
        current_time = time.time() / 60  # Convert to minutes from epoch start (for relative timing)
        
        # Set T: All trains
        self.T = set(self.trains_data.keys())
        
        # Set B: All segments  
        self.B = set(self.tracks.keys())
        
        # Initialize per-segment track sets P_j and alternatives R_j
        for segment_id, segment in self.tracks.items():
            self.P[segment_id] = list(range(1, segment.num_tracks + 1))
            self.R[segment_id] = segment.alternative_segments or []
        
        # Create enhanced events with through destinations
        self._create_enhanced_events()
        
        # Initialize enhanced cost parameters
        for train_id in self.T:
            train = self.trains_data[train_id]
            priority = train.get('priority', 'medium')
            train_type = train.get('train_type', 'passenger')
            
            # Cost per minute of delay (paper's c^low_i)
            if priority == 'high':
                self.c_low[train_id] = 20.0
            elif priority == 'medium':
                self.c_low[train_id] = 10.0
            else:  # low priority
                self.c_low[train_id] = 5.0
                
            # Higher costs for passenger trains
            if train_type == 'passenger':
                self.c_low[train_id] *= 2.0
                
            # Penalty threshold and cost (paper's w_i and c^penalty_i)
            self.w[train_id] = 60.0 if train_type == 'passenger' else 120.0  # 1-2 hour threshold
            self.c_penalty[train_id] = 500.0 if priority == 'high' else 200.0  # High penalty for excessive delays
            
            # Rerouting cost (should be significant but not prohibitive)
            self.c_reroute[train_id] = 50.0 if priority == 'high' else 20.0
            
            # Train length (paper's g^train_i)
            if train_type == 'freight':
                self.g_train[train_id] = 800.0  # meters (longer freight trains)
            else:
                self.g_train[train_id] = 400.0  # meters (passenger trains)
    
    def _create_enhanced_events(self):
        """Create enhanced events with through destination support"""
        self.E = set()
        self.K = {train_id: [] for train_id in self.T}
        self.L = {segment_id: [] for segment_id in self.B}
        
        for train_id, train in self.trains_data.items():
            stops = train.get('stops', [])
            if isinstance(stops, str):
                try:
                    stops = ast.literal_eval(stops)  # Safer than eval
                except:
                    stops = [stops]  # Single stop case
                    
            thru_dest = train.get('thru_dest', '')
            
            # Extend stops for through destinations
            extended_stops = list(stops)
            if thru_dest and thru_dest.strip():
                if thru_dest == 'Erode':
                    extended_stops.append('MYS_Erode')
                elif thru_dest == 'Chennai':
                    extended_stops.append('MYS_Chennai')
            
            dep_time = train.get('dep_time', 0)  # Initial departure time in minutes
            current_time = dep_time
            
            # Create events for this train's journey
            train_events = []
            
            for i, station in enumerate(extended_stops):
                # Handle both original stations and through destinations
                segment_id = station
                if station not in self.tracks:
                    # Skip if segment doesn't exist
                    logger.warning(f"Segment {station} not found for train {train_id}")
                    continue
                
                event_id = f"{train_id}_{segment_id}_{i}"
                
                # Calculate planned times
                if i == 0:
                    # First station (origin)
                    planned_arrival = dep_time
                    planned_departure = dep_time
                    is_ongoing = current_time <= dep_time + 5  # If within 5 minutes, consider ongoing
                else:
                    # Calculate travel time from previous valid station
                    prev_station = None
                    for j in range(i-1, -1, -1):
                        if extended_stops[j] in self.tracks:
                            prev_station = extended_stops[j]
                            break
                    
                    if prev_station:
                        travel_time = self._calculate_enhanced_travel_time(train_id, prev_station, station)
                        planned_arrival = current_time + travel_time
                    else:
                        planned_arrival = current_time + 30  # Default 30 minutes
                        
                    # Add dwell time
                    dwell_time = self._get_enhanced_dwell_time(train_id, station)
                    planned_departure = planned_arrival + dwell_time
                
                current_time = planned_departure
                
                # Determine if this is a planned stop
                is_planned_stop = station in stops or station.endswith('_Erode') or station.endswith('_Chennai')
                
                # Create enhanced TrainEvent
                event = TrainEvent(
                    event_id=event_id,
                    train_id=train_id,
                    segment_id=segment_id,
                    is_station=self.tracks[segment_id].is_station,
                    planned_arrival=planned_arrival,
                    planned_departure=planned_departure,
                    min_duration=dwell_time if i > 0 else 0,
                    is_planned_stop=is_planned_stop,
                    origin_direction=0 if i < len(extended_stops)//2 else 1,
                    is_ongoing=is_ongoing if i == 0 else False,
                    is_forced=is_ongoing if i == 0 else False
                )
                
                train_events.append(event)
                self.E.add(event_id)
                
                # Add to segment event list
                if segment_id not in self.L:
                    self.L[segment_id] = []
                self.L[segment_id].append(event_id)
                
                # Create line segment events between stations
                if i < len(extended_stops) - 1:
                    next_station = extended_stops[i + 1]
                    if next_station in self.tracks:
                        line_event = self._create_line_event(train_id, station, next_station, current_time, i)
                        if line_event:
                            train_events.append(line_event)
                            self.E.add(line_event.event_id)
                            if line_event.segment_id not in self.L:
                                self.L[line_event.segment_id] = []
                            self.L[line_event.segment_id].append(line_event.event_id)
                            current_time = line_event.planned_departure
            
            self.K[train_id] = train_events
            logger.info(f"Created {len(train_events)} events for train {train_id}")
    
    def _create_line_event(self, train_id: str, from_station: str, to_station: str, start_time: float, sequence: int) -> Optional[TrainEvent]:
        """Create line segment event between stations"""
        # Find appropriate line segment
        line_segment_id = None
        
        # Try direct segment names
        possible_segments = [
            f"{from_station}_{to_station}",
            f"{to_station}_{from_station}",
        ]
        
        for seg_id in possible_segments:
            if seg_id in self.tracks:
                line_segment_id = seg_id
                break
        
        if not line_segment_id:
            logger.warning(f"No line segment found between {from_station} and {to_station}")
            return None
        
        travel_time = self._calculate_enhanced_travel_time(train_id, from_station, to_station)
        event_id = f"{train_id}_{line_segment_id}_{sequence}"
        
        return TrainEvent(
            event_id=event_id,
            train_id=train_id,
            segment_id=line_segment_id,
            is_station=False,
            planned_arrival=start_time,
            planned_departure=start_time + travel_time,
            min_duration=travel_time,
            is_planned_stop=False,
            origin_direction=0,
            is_ongoing=False,
            is_forced=False
        )
    
    def _calculate_enhanced_travel_time(self, train_id: str, from_station: str, to_station: str) -> float:
        """Calculate enhanced travel time with real distances"""
        train = self.trains_data[train_id]
        speed_kmh = train.get('speed_kmh', 60)
        
        # Use real distances if available
        distance_km = self._get_real_distance(from_station, to_station)
        
        # Calculate base travel time
        travel_time = (distance_km / speed_kmh) * 60  # minutes
        
        # Add speed restrictions for freight trains
        if train.get('train_type') == 'freight':
            travel_time *= 1.2  # 20% slower
        
        # Add buffer for complex routes
        if distance_km > 100:  # Long distance
            travel_time += 5  # 5 minute buffer
        
        return max(travel_time, 5.0)  # Minimum 5 minutes
    
    def _get_real_distance(self, from_station: str, to_station: str) -> float:
        """Get real distance between stations using coordinates"""
        # Check cache first
        cache_key = f"{from_station}_{to_station}"
        if cache_key in self.distance_matrix:
            return self.distance_matrix[cache_key]
        
        # Default distances (fallback)
        default_distances = {
            ('SBC', 'Kengeri'): 25.0,
            ('Kengeri', 'MYA'): 60.0,
            ('MYA', 'Channapatna'): 45.0,
            ('Channapatna', 'MYS'): 55.0,
            ('MYS', 'MYS_Erode'): 120.0,
            ('MYS', 'MYS_Chennai'): 180.0,
        }
        
        # Try both directions
        distance = (default_distances.get((from_station, to_station)) or 
                   default_distances.get((to_station, from_station)) or 
                   30.0)  # Default 30km
        
        # Try to get real distance if coordinates available
        try:
            from_coords = None
            to_coords = None
            
            if from_station in self.tracks and self.tracks[from_station].coordinates:
                from_coords = self.tracks[from_station].coordinates[0]
            if to_station in self.tracks and self.tracks[to_station].coordinates:
                to_coords = self.tracks[to_station].coordinates[0]
            
            if from_coords and to_coords:
                real_distance = geodesic(
                    (from_coords[1], from_coords[0]),  # lat, lon
                    (to_coords[1], to_coords[0])
                ).kilometers
                distance = real_distance
                logger.debug(f"Real distance {from_station}-{to_station}: {distance:.1f} km")
        except Exception as e:
            logger.debug(f"Could not calculate real distance for {from_station}-{to_station}: {e}")
        
        # Cache the result
        self.distance_matrix[cache_key] = distance
        return distance
    
    def _get_enhanced_dwell_time(self, train_id: str, station: str) -> float:
        """Get enhanced dwell time at station"""
        train = self.trains_data[train_id]
        train_type = train.get('train_type', 'passenger')
        priority = train.get('priority', 'medium')
        
        # Base dwell times by station type and train type
        if station.endswith('_Erode') or station.endswith('_Chennai'):
            # Through destinations have longer times
            base_time = 10.0 if train_type == 'passenger' else 20.0
        elif station in ['SBC', 'MYS']:
            # Terminal stations
            base_time = 8.0 if train_type == 'passenger' else 15.0
        elif station in ['MYA']:
            # Major junction
            base_time = 5.0 if train_type == 'passenger' else 12.0
        else:
            # Regular stations
            base_time = 2.0 if train_type == 'passenger' else 8.0
        
        # Adjust for priority
        if priority == 'high':
            base_time *= 0.8  # High priority stops less
        elif priority == 'low':
            base_time *= 1.3  # Low priority waits longer
        
        return max(base_time, 1.0)  # Minimum 1 minute
    
    def optimize_schedule(self, disrupted_trains: List[str] = None, 
                         optimization_type: str = "total_delay",
                         strategy: int = 3,
                         enable_rerouting: bool = True) -> OptimizationResult:
        """
        Enhanced MILP optimization with rerouting and through destination support
        """
        start_time = time.time()
        logger.info(f"Starting enhanced MILP optimization (strategy {strategy}, rerouting: {enable_rerouting})")
        
        try:
            # Validate input data
            if not self._validate_model_data():
                raise ValueError("Model data validation failed")
            
            # Create PuLP problem
            if optimization_type == "total_delay":
                prob = pulp.LpProblem("Enhanced_Railway_Rescheduling_MinDelay", pulp.LpMinimize)
            else:
                prob = pulp.LpProblem("Enhanced_Railway_Rescheduling_WeightedCost", pulp.LpMinimize)
            
            # Enhanced decision variables
            x_begin = {}  # x^begin_k: Start time of event k
            x_end = {}    # x^end_k: End time of event k
            z = {}        # z_k: Delay of event k
            z_final = {}  # z_n_i: Final delay of train i
            x_order = {}  # x_{k,k'}: Binary ordering variable
            y_track = {}  # y_{k,p}: Binary track assignment
            e_penalty = {}  # e_i: Binary penalty activation
            r_reroute = {}  # r_{k,alt}: Binary rerouting variables
            
            # Create enhanced decision variables
            self._create_enhanced_variables(prob, x_begin, x_end, z, z_final, 
                                          x_order, y_track, e_penalty, r_reroute, 
                                          strategy, enable_rerouting)
            
            # Add enhanced constraints
            self._add_enhanced_constraints(prob, x_begin, x_end, z, z_final,
                                         x_order, y_track, e_penalty, r_reroute,
                                         strategy, enable_rerouting)
            
            # Set enhanced objective function
            self._set_enhanced_objective(prob, z_final, e_penalty, r_reroute, 
                                       optimization_type, enable_rerouting)
            
            # Solve with enhanced solver settings
            solver = pulp.PULP_CBC_CMD(
                timeLimit=self.max_optimization_time,
                msg=True,
                gapRel=0.05,  # 5% optimality gap acceptable
                threads=4     # Use multiple threads
            )
            
            logger.info("Solving MILP problem...")
            solve_status = prob.solve(solver)
            
            # Process enhanced results
            result = self._process_enhanced_results(
                prob, x_begin, x_end, z_final, y_track, x_order, r_reroute,
                start_time, optimization_type, strategy, enable_rerouting
            )
            
            logger.info(f"MILP optimization completed: {result.status} in {result.computation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced MILP optimization failed: {e}", exc_info=True)
            return OptimizationResult(
                status="failed",
                objective_value=float('inf'),
                total_delay=0.0,
                delay_reduction=0.0,
                decisions=[],
                optimized_times={},
                track_assignments={},
                order_changes=[],
                rerouting_decisions=[],
                computation_time=time.time() - start_time,
                solver_status=str(e)
            )
    
    def _validate_model_data(self) -> bool:
        """Validate MILP model data before optimization"""
        try:
            # Check trains
            if not self.T:
                logger.error("No trains in the model")
                return False
            
            # Check segments
            if not self.B:
                logger.error("No segments in the model")
                return False
            
            # Check events
            if not self.E:
                logger.error("No events in the model")
                return False
            
            # Validate events have valid segments
            for event_id in self.E:
                event = self._get_event_by_id(event_id)
                if event.segment_id not in self.tracks:
                    logger.error(f"Event {event_id} references invalid segment {event.segment_id}")
                    return False
            
            # Validate track assignments
            for segment_id in self.B:
                if segment_id not in self.P or not self.P[segment_id]:
                    logger.error(f"Segment {segment_id} has no tracks")
                    return False
            
            logger.info("Model data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _create_enhanced_variables(self, prob, x_begin, x_end, z, z_final, 
                                 x_order, y_track, e_penalty, r_reroute, 
                                 strategy, enable_rerouting):
        """Create enhanced MILP decision variables"""
        logger.info("Creating enhanced decision variables...")
        
        # Time variables for events
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            
            # Lower bounds based on current time and constraints
            min_start = max(0, event.planned_arrival - 30)  # Allow 30 min earlier
            max_end = event.planned_departure + 180  # Allow up to 3 hours delay
            
            x_begin[event_id] = pulp.LpVariable(f"x_begin_{event_id}", 
                                              lowBound=min_start, upBound=max_end)
            x_end[event_id] = pulp.LpVariable(f"x_end_{event_id}", 
                                            lowBound=min_start, upBound=max_end)
            z[event_id] = pulp.LpVariable(f"z_{event_id}", lowBound=0, upBound=180)
        
        # Final delay and penalty variables for trains
        for train_id in self.T:
            z_final[train_id] = pulp.LpVariable(f"z_final_{train_id}", lowBound=0, upBound=300)
            e_penalty[train_id] = pulp.LpVariable(f"e_penalty_{train_id}", cat='Binary')
        
        # Track assignment variables
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            segment_id = event.segment_id
            
            if segment_id in self.P:
                for track in self.P[segment_id]:
                    var_name = f"y_track_{event_id}_{track}"
                    y_track[(event_id, track)] = pulp.LpVariable(var_name, cat='Binary')
        
        # Ordering variables (strategy-dependent)
        if strategy >= 2:
            for segment_id in self.B:
                events_on_segment = self.L.get(segment_id, [])
                for i, event1 in enumerate(events_on_segment):
                    for j, event2 in enumerate(events_on_segment):
                        if i != j:
                            var_name = f"x_order_{event1}_{event2}"
                            x_order[(event1, event2)] = pulp.LpVariable(var_name, cat='Binary')
        
        # Rerouting variables (if enabled)
        if enable_rerouting:
            for event_id in self.E:
                event = self._get_event_by_id(event_id)
                segment_id = event.segment_id
                
                if segment_id in self.R and self.R[segment_id]:
                    for alt_segment in self.R[segment_id]:
                        if alt_segment in self.tracks:
                            var_name = f"r_reroute_{event_id}_{alt_segment}"
                            r_reroute[(event_id, alt_segment)] = pulp.LpVariable(var_name, cat='Binary')
        
        logger.info(f"Created variables: {len(x_begin)} time vars, {len(y_track)} track vars, "
                   f"{len(x_order)} order vars, {len(r_reroute)} reroute vars")
    
    def _add_enhanced_constraints(self, prob, x_begin, x_end, z, z_final,
                                x_order, y_track, e_penalty, r_reroute,
                                strategy, enable_rerouting):
        """Add enhanced MILP constraints"""
        logger.info("Adding enhanced constraints...")
        
        # 1. Sequencing constraints (enhanced)
        self._add_enhanced_sequencing_constraints(prob, x_begin, x_end)
        
        # 2. Duration constraints (enhanced)
        self._add_enhanced_duration_constraints(prob, x_begin, x_end, r_reroute, enable_rerouting)
        
        # 3. Planned stop constraints
        self._add_enhanced_stop_constraints(prob, x_begin)
        
        # 4. Delay recording constraints
        self._add_enhanced_delay_constraints(prob, x_end, z, z_final)
        
        # 5. Track assignment constraints (enhanced)
        self._add_enhanced_track_constraints(prob, y_track, r_reroute, enable_rerouting)
        
        # 6. Separation constraints (enhanced)
        self._add_enhanced_separation_constraints(prob, x_begin, x_end, x_order, y_track, strategy)
        
        # 7. Train length constraints
        self._add_enhanced_length_constraints(prob, y_track)
        
        # 8. Penalty constraints
        self._add_enhanced_penalty_constraints(prob, z_final, e_penalty)
        
        # 9. Rerouting constraints (new)
        if enable_rerouting:
            self._add_rerouting_constraints(prob, r_reroute, y_track)
        
        # 10. Fixed event constraints (for ongoing operations)
        self._add_fixed_event_constraints(prob, x_begin, x_end)
        
        logger.info("Enhanced constraints added successfully")
    
    def _add_enhanced_sequencing_constraints(self, prob, x_begin, x_end):
        """Enhanced sequencing constraints with validation"""
        constraint_count = 0
        for train_id in self.T:
            events = self.K[train_id]
            for i in range(len(events) - 1):
                current_event = events[i]
                next_event = events[i + 1]
                
                # Only add constraint if both events have variables
                if (current_event.event_id in x_end and 
                    next_event.event_id in x_begin):
                    
                    constraint_name = f"seq_{train_id}_{i}"
                    prob += (x_end[current_event.event_id] <= 
                           x_begin[next_event.event_id], constraint_name)
                    constraint_count += 1
        
        logger.info(f"Added {constraint_count} sequencing constraints")
    
    def _add_enhanced_duration_constraints(self, prob, x_begin, x_end, r_reroute, enable_rerouting):
        """Enhanced duration constraints with rerouting support"""
        constraint_count = 0
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            
            if event_id in x_begin and event_id in x_end:
                base_duration = event.min_duration
                
                # Base duration constraint
                constraint_name = f"dur_{event_id}"
                prob += (x_end[event_id] >= x_begin[event_id] + base_duration, constraint_name)
                constraint_count += 1
                
                # Additional duration for rerouting
                if enable_rerouting and event.segment_id in self.R:
                    for alt_segment in self.R[event.segment_id]:
                        if (event_id, alt_segment) in r_reroute:
                            # Rerouting typically takes longer
                            alt_duration = base_duration * 1.2  # 20% longer
                            constraint_name = f"dur_alt_{event_id}_{alt_segment}"
                            prob += (x_end[event_id] >= x_begin[event_id] + alt_duration - 
                                   self.M * (1 - r_reroute[(event_id, alt_segment)]), 
                                   constraint_name)
                            constraint_count += 1
        
        logger.info(f"Added {constraint_count} duration constraints")
    
    def _add_enhanced_stop_constraints(self, prob, x_begin):
        """Enhanced planned stop constraints"""
        constraint_count = 0
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            
            if event.is_planned_stop and not event.is_forced and event_id in x_begin:
                # Cannot depart earlier than planned (with small tolerance)
                earliest_departure = event.planned_departure - 5  # 5 minute early tolerance
                constraint_name = f"stop_{event_id}"
                prob += (x_begin[event_id] >= earliest_departure, constraint_name)
                constraint_count += 1
        
        logger.info(f"Added {constraint_count} planned stop constraints")
    
    def _add_enhanced_delay_constraints(self, prob, x_end, z, z_final):
        """Enhanced delay recording constraints"""
        constraint_count = 0
        
        # Delay recording for each event
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            
            if event_id in x_end and event_id in z:
                constraint_name = f"delay_{event_id}"
                prob += (z[event_id] >= x_end[event_id] - event.planned_departure, constraint_name)
                constraint_count += 1
        
        # Final delay is maximum delay of train's events
        for train_id in self.T:
            events = self.K[train_id]
            if events and train_id in z_final:
                # Final delay is the delay of the last event
                last_event = events[-1]
                if last_event.event_id in z:
                    constraint_name = f"final_delay_{train_id}"
                    prob += (z_final[train_id] >= z[last_event.event_id], constraint_name)
                    constraint_count += 1
        
        logger.info(f"Added {constraint_count} delay constraints")
    
    def _add_enhanced_track_constraints(self, prob, y_track, r_reroute, enable_rerouting):
        """Enhanced track assignment constraints"""
        constraint_count = 0
        
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            segment_id = event.segment_id
            
            if segment_id in self.P:
                # Each event must be assigned to exactly one track (original or rerouted)
                track_vars = [y_track.get((event_id, track), 0) for track in self.P[segment_id]]
                
                # Add rerouting alternatives
                if enable_rerouting and segment_id in self.R:
                    for alt_segment in self.R[segment_id]:
                        if alt_segment in self.P:
                            for track in self.P[alt_segment]:
                                if (event_id, alt_segment) in r_reroute:
                                    # Track assignment on alternative segment
                                    track_vars.append(r_reroute[(event_id, alt_segment)])
                
                if track_vars:
                    constraint_name = f"track_assign_{event_id}"
                    prob += (pulp.lpSum(track_vars) == 1, constraint_name)
                    constraint_count += 1
        
        logger.info(f"Added {constraint_count} track assignment constraints")
    
    def _add_enhanced_separation_constraints(self, prob, x_begin, x_end, x_order, y_track, strategy):
        """Enhanced separation constraints with improved conflict handling"""
        constraint_count = 0
        
        for segment_id in self.B:
            segment = self.tracks[segment_id]
            events_on_segment = self.L.get(segment_id, [])
            
            for i, event1_id in enumerate(events_on_segment):
                for j, event2_id in enumerate(events_on_segment):
                    if i != j:
                        event1 = self._get_event_by_id(event1_id)
                        event2 = self._get_event_by_id(event2_id)
                        
                        # Check for conflicts on each track
                        for track in self.P[segment_id]:
                            if ((event1_id, track) in y_track and 
                                (event2_id, track) in y_track):
                                
                                # Determine separation time
                                if event1.origin_direction != event2.origin_direction:
                                    sep_time = segment.meet_separation
                                else:
                                    sep_time = segment.follow_separation
                                
                                # Add separation constraints
                                if strategy >= 2 and (event1_id, event2_id) in x_order:
                                    # With ordering variables
                                    constraint_name = f"sep_{event1_id}_{event2_id}_{track}_1"
                                    prob += (x_begin[event2_id] >= x_end[event1_id] + sep_time - 
                                           self.M * (3 - y_track[(event1_id, track)] - 
                                                   y_track[(event2_id, track)] - 
                                                   x_order[(event1_id, event2_id)]), 
                                           constraint_name)
                                    
                                    constraint_name = f"sep_{event1_id}_{event2_id}_{track}_2"
                                    prob += (x_begin[event1_id] >= x_end[event2_id] + sep_time - 
                                           self.M * (2 - y_track[(event1_id, track)] - 
                                                   y_track[(event2_id, track)] + 
                                                   x_order[(event1_id, event2_id)]), 
                                           constraint_name)
                                    constraint_count += 2
                                    
                                else:
                                    # Without reordering (fixed order)
                                    if event1.planned_departure <= event2.planned_departure:
                                        constraint_name = f"sep_fixed_{event1_id}_{event2_id}_{track}"
                                        prob += (x_begin[event2_id] >= x_end[event1_id] + sep_time - 
                                               self.M * (2 - y_track[(event1_id, track)] - 
                                                       y_track[(event2_id, track)]), 
                                               constraint_name)
                                        constraint_count += 1
        
        # Mutual exclusion for ordering variables
        if strategy >= 2:
            for segment_id in self.B:
                events_on_segment = self.L.get(segment_id, [])
                for i, event1_id in enumerate(events_on_segment):
                    for j, event2_id in enumerate(events_on_segment):
                        if (i != j and (event1_id, event2_id) in x_order and 
                            (event2_id, event1_id) in x_order):
                            constraint_name = f"mutex_{event1_id}_{event2_id}"
                            prob += (x_order[(event1_id, event2_id)] + 
                                   x_order[(event2_id, event1_id)] <= 1, constraint_name)
                            constraint_count += 1
        
        logger.info(f"Added {constraint_count} separation constraints")
    
    def _add_enhanced_length_constraints(self, prob, y_track):
        """Enhanced train length constraints"""
        constraint_count = 0
        
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            segment_id = event.segment_id
            train_id = event.train_id
            
            if segment_id in self.P:
                segment = self.tracks[segment_id]
                train_length = self.g_train[train_id]
                
                for track_num, track in enumerate(self.P[segment_id]):
                    if (event_id, track) in y_track:
                        if track_num < len(segment.track_lengths):
                            track_length = segment.track_lengths[track_num]
                            
                            constraint_name = f"length_{event_id}_{track}"
                            prob += (train_length * y_track[(event_id, track)] <= track_length, 
                                   constraint_name)
                            constraint_count += 1
        
        logger.info(f"Added {constraint_count} train length constraints")
    
    def _add_enhanced_penalty_constraints(self, prob, z_final, e_penalty):
        """Enhanced penalty constraints"""
        constraint_count = 0
        
        for train_id in self.T:
            if train_id in z_final and train_id in e_penalty:
                threshold = self.w[train_id]
                constraint_name = f"penalty_{train_id}"
                prob += (z_final[train_id] - threshold <= self.M * e_penalty[train_id], 
                       constraint_name)
                constraint_count += 1
        
        logger.info(f"Added {constraint_count} penalty constraints")
    
    def _add_rerouting_constraints(self, prob, r_reroute, y_track):
        """Add rerouting-specific constraints"""
        constraint_count = 0
        
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            segment_id = event.segment_id
            
            # Rerouting mutual exclusion
            if segment_id in self.R and self.R[segment_id]:
                reroute_vars = []
                for alt_segment in self.R[segment_id]:
                    if (event_id, alt_segment) in r_reroute:
                        reroute_vars.append(r_reroute[(event_id, alt_segment)])
                
                if reroute_vars:
                    # At most one rerouting per event
                    constraint_name = f"reroute_mutex_{event_id}"
                    prob += (pulp.lpSum(reroute_vars) <= 1, constraint_name)
                    constraint_count += 1
        
        logger.info(f"Added {constraint_count} rerouting constraints")
    
    def _add_fixed_event_constraints(self, prob, x_begin, x_end):
        """Add constraints for fixed/ongoing events"""
        constraint_count = 0
        
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            
            if event.is_forced:
                # Fixed events cannot change times
                if event_id in x_begin:
                    constraint_name = f"fixed_begin_{event_id}"
                    prob += (x_begin[event_id] == event.planned_departure, constraint_name)
                    constraint_count += 1
                
                if event_id in x_end:
                    constraint_name = f"fixed_end_{event_id}"
                    prob += (x_end[event_id] == event.planned_arrival, constraint_name)
                    constraint_count += 1
        
        logger.info(f"Added {constraint_count} fixed event constraints")
    
    def _set_enhanced_objective(self, prob, z_final, e_penalty, r_reroute, 
                              optimization_type, enable_rerouting):
        """Set enhanced objective function"""
        if optimization_type == "total_delay":
            # Objective 1a: Minimize total final delay
            objective_terms = [z_final[train_id] for train_id in self.T if train_id in z_final]
            
        else:
            # Objective 1b: Minimize weighted costs
            objective_terms = []
            for train_id in self.T:
                if train_id in z_final:
                    # Delay cost
                    objective_terms.append(self.c_low[train_id] * z_final[train_id])
                    
                if train_id in e_penalty:
                    # Penalty cost
                    objective_terms.append(self.c_penalty[train_id] * e_penalty[train_id])
        
        # Add rerouting costs
        if enable_rerouting:
            for (event_id, alt_segment), var in r_reroute.items():
                event = self._get_event_by_id(event_id)
                train_id = event.train_id
                reroute_cost = self.c_reroute[train_id]
                objective_terms.append(reroute_cost * var)
        
        if objective_terms:
            prob += pulp.lpSum(objective_terms)
            logger.info(f"Set objective with {len(objective_terms)} terms")
        else:
            logger.warning("No objective terms found, using dummy objective")
            prob += 0
    
    def _get_event_by_id(self, event_id: str) -> TrainEvent:
        """Get TrainEvent object by event_id with error handling"""
        for train_id in self.T:
            for event in self.K[train_id]:
                if event.event_id == event_id:
                    return event
        
        # If not found, create a dummy event to prevent crashes
        logger.warning(f"Event {event_id} not found, creating dummy event")
        return TrainEvent(
            event_id=event_id,
            train_id="unknown",
            segment_id="unknown",
            is_station=True,
            planned_arrival=0,
            planned_departure=0,
            min_duration=5,
            is_planned_stop=False,
            origin_direction=0
        )
    
    def _process_enhanced_results(self, prob, x_begin, x_end, z_final, y_track, 
                                x_order, r_reroute, start_time, optimization_type, 
                                strategy, enable_rerouting) -> OptimizationResult:
        """Process enhanced MILP optimization results"""
        computation_time = time.time() - start_time
        status_mapping = {
            pulp.LpStatusOptimal: "optimal",
            pulp.LpStatusInfeasible: "infeasible",
            pulp.LpStatusUnbounded: "unbounded",
            pulp.LpStatusNotSolved: "not_solved",
            pulp.LpStatusUndefined: "undefined"
        }
        
        status = status_mapping.get(prob.status, "unknown")
        
        if status != "optimal":
            logger.warning(f"Optimization not optimal: {status}")
            return OptimizationResult(
                status=status,
                objective_value=float('inf') if status == "infeasible" else 0.0,
                total_delay=0.0,
                delay_reduction=0.0,
                decisions=[],
                optimized_times={},
                track_assignments={},
                order_changes=[],
                rerouting_decisions=[],
                computation_time=computation_time,
                solver_status=pulp.LpStatus[prob.status]
            )
        
        # Extract solution values
        objective_value = pulp.value(prob.objective) or 0.0
        
        # Calculate delays and improvements
        total_original_delay = 0.0
        total_optimized_delay = 0.0
        
        optimized_times = {}
        track_assignments = {}
        order_changes = []
        rerouting_decisions = []
        decisions = []
        
        try:
            # Process event times and delays
            for event_id in self.E:
                event = self._get_event_by_id(event_id)
                
                # Get optimized times
                begin_time = pulp.value(x_begin.get(event_id, event.planned_departure))
                end_time = pulp.value(x_end.get(event_id, event.planned_arrival))
                
                if begin_time is not None and end_time is not None:
                    optimized_times[event_id] = {
                        'planned_arrival': event.planned_arrival,
                        'planned_departure': event.planned_departure,
                        'optimized_arrival': float(end_time),
                        'optimized_departure': float(begin_time),
                        'delay': max(0, float(end_time) - event.planned_arrival)
                    }
                    
                    # Track total delays
                    original_delay = max(0, event.planned_arrival - event.planned_arrival)
                    optimized_delay = max(0, float(end_time) - event.planned_arrival)
                    total_original_delay += original_delay
                    total_optimized_delay += optimized_delay
            
            # Process track assignments
            for (event_id, track), var in y_track.items():
                if pulp.value(var) and pulp.value(var) > 0.5:
                    event = self._get_event_by_id(event_id)
                    track_assignments[event_id] = {
                        'train_id': event.train_id,
                        'segment_id': event.segment_id,
                        'track': track,
                        'confidence': float(pulp.value(var))
                    }
            
            # Process order changes
            if strategy >= 2:
                for (event1_id, event2_id), var in x_order.items():
                    if pulp.value(var) and pulp.value(var) > 0.5:
                        event1 = self._get_event_by_id(event1_id)
                        event2 = self._get_event_by_id(event2_id)
                        
                        # Check if this is actually a reordering
                        original_order = event1.planned_departure <= event2.planned_departure
                        new_order = True  # event1 before event2 in optimized solution
                        
                        if original_order != new_order:
                            order_changes.append({
                                'event1_id': event1_id,
                                'event2_id': event2_id,
                                'train1_id': event1.train_id,
                                'train2_id': event2.train_id,
                                'segment_id': event1.segment_id,
                                'original_order': original_order,
                                'new_order': new_order
                            })
            
            # Process rerouting decisions
            if enable_rerouting:
                for (event_id, alt_segment), var in r_reroute.items():
                    if pulp.value(var) and pulp.value(var) > 0.5:
                        event = self._get_event_by_id(event_id)
                        rerouting_decisions.append({
                            'event_id': event_id,
                            'train_id': event.train_id,
                            'original_segment': event.segment_id,
                            'alternative_segment': alt_segment,
                            'confidence': float(pulp.value(var))
                        })
            
            # Create optimization decisions
            decisions = self._create_enhanced_decisions(
                optimized_times, track_assignments, order_changes, rerouting_decisions
            )
            
            # Calculate improvement metrics
            delay_reduction = max(0, total_original_delay - total_optimized_delay)
            delay_reduction_percent = (delay_reduction / max(total_original_delay, 1)) * 100
            
            logger.info(f"Optimization completed successfully:")
            logger.info(f"  Objective value: {objective_value:.2f}")
            logger.info(f"  Total delay: {total_optimized_delay:.2f} min")
            logger.info(f"  Delay reduction: {delay_reduction:.2f} min ({delay_reduction_percent:.1f}%)")
            logger.info(f"  Decisions generated: {len(decisions)}")
            logger.info(f"  Track assignments: {len(track_assignments)}")
            logger.info(f"  Order changes: {len(order_changes)}")
            logger.info(f"  Rerouting decisions: {len(rerouting_decisions)}")
            
            return OptimizationResult(
                status="optimal",
                objective_value=objective_value,
                total_delay=total_optimized_delay,
                delay_reduction=delay_reduction,
                decisions=decisions,
                optimized_times=optimized_times,
                track_assignments=track_assignments,
                order_changes=order_changes,
                rerouting_decisions=rerouting_decisions,
                computation_time=computation_time,
                solver_status=pulp.LpStatus[prob.status]
            )
            
        except Exception as e:
            logger.error(f"Error processing optimization results: {e}", exc_info=True)
            return OptimizationResult(
                status="error",
                objective_value=objective_value,
                total_delay=0.0,
                delay_reduction=0.0,
                decisions=[],
                optimized_times={},
                track_assignments={},
                order_changes=[],
                rerouting_decisions=[],
                computation_time=computation_time,
                solver_status=f"Error: {str(e)}"
            )
    
    def _create_enhanced_decisions(self, optimized_times, track_assignments, 
                                 order_changes, rerouting_decisions) -> List[OptimizationDecision]:
        """Create enhanced optimization decisions"""
        decisions = []
        decision_id = 1
        
        # Time adjustment decisions
        for event_id, times in optimized_times.items():
            if abs(times['optimized_departure'] - times['planned_departure']) > 1.0:
                event = self._get_event_by_id(event_id)
                
                decisions.append(OptimizationDecision(
                    decision_id=f"TIME_{decision_id}",
                    decision_type="time_adjustment",
                    train_id=event.train_id,
                    event_id=event_id,
                    segment_id=event.segment_id,
                    description=f"Adjust departure time from {times['planned_departure']:.1f} "
                              f"to {times['optimized_departure']:.1f} (delay: {times['delay']:.1f} min)",
                    impact_score=times['delay'],
                    original_value=times['planned_departure'],
                    new_value=times['optimized_departure'],
                    confidence=0.9,
                    status="pending",
                    created_at=datetime.now()
                ))
                decision_id += 1
        
        # Track assignment decisions
        for event_id, assignment in track_assignments.items():
            event = self._get_event_by_id(event_id)
            
            decisions.append(OptimizationDecision(
                decision_id=f"TRACK_{decision_id}",
                decision_type="track_assignment",
                train_id=assignment['train_id'],
                event_id=event_id,
                segment_id=assignment['segment_id'],
                description=f"Assign train {assignment['train_id']} to track {assignment['track']} "
                          f"at segment {assignment['segment_id']}",
                impact_score=5.0,  # Standard track assignment impact
                original_value=None,
                new_value=assignment['track'],
                confidence=assignment['confidence'],
                status="pending",
                created_at=datetime.now()
            ))
            decision_id += 1
        
        # Order change decisions
        for order_change in order_changes:
            decisions.append(OptimizationDecision(
                decision_id=f"ORDER_{decision_id}",
                decision_type="order_change",
                train_id=order_change['train1_id'],
                event_id=order_change['event1_id'],
                segment_id=order_change['segment_id'],
                description=f"Reorder trains {order_change['train1_id']} and {order_change['train2_id']} "
                          f"at segment {order_change['segment_id']}",
                impact_score=10.0,  # Higher impact for reordering
                original_value=order_change['original_order'],
                new_value=order_change['new_order'],
                confidence=0.8,
                status="pending",
                created_at=datetime.now()
            ))
            decision_id += 1
        
        # Rerouting decisions
        for reroute in rerouting_decisions:
            decisions.append(OptimizationDecision(
                decision_id=f"REROUTE_{decision_id}",
                decision_type="rerouting",
                train_id=reroute['train_id'],
                event_id=reroute['event_id'],
                segment_id=reroute['original_segment'],
                description=f"Reroute train {reroute['train_id']} from segment "
                          f"{reroute['original_segment']} to {reroute['alternative_segment']}",
                impact_score=15.0,  # High impact for rerouting
                original_value=reroute['original_segment'],
                new_value=reroute['alternative_segment'],
                confidence=reroute['confidence'],
                status="pending",
                created_at=datetime.now()
            ))
            decision_id += 1
        
        return decisions
    
    def get_optimization_summary(self, result: OptimizationResult) -> dict:
        """Get enhanced optimization summary"""
        summary = {
            'status': result.status,
            'computation_time': result.computation_time,
            'objective_value': result.objective_value,
            'total_delay': result.total_delay,
            'delay_reduction': result.delay_reduction,
            'decisions_count': len(result.decisions),
            'track_assignments_count': len(result.track_assignments),
            'order_changes_count': len(result.order_changes),
            'rerouting_decisions_count': len(result.rerouting_decisions),
            'solver_status': result.solver_status
        }
        
        # Add decision type breakdown
        decision_types = {}
        for decision in result.decisions:
            decision_types[decision.decision_type] = decision_types.get(decision.decision_type, 0) + 1
        summary['decision_types'] = decision_types
        
        # Add performance metrics
        if result.total_delay > 0:
            summary['delay_reduction_percent'] = (result.delay_reduction / result.total_delay) * 100
        else:
            summary['delay_reduction_percent'] = 0.0
        
        # Add model statistics
        summary['model_stats'] = {
            'trains': len(self.T),
            'segments': len(self.B),
            'events': len(self.E),
            'tracks': sum(len(tracks) for tracks in self.P.values()),
            'alternative_routes': sum(len(alts) for alts in self.R.values())
        }
        
        return summary