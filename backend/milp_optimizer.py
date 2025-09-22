"""
Enhanced MILP Optimizer for Railway Decision Support System
Based on Törnquist & Persson (2007) "N-tracked Railway Traffic Re-scheduling During Disturbances"
Implements event-based MILP formulation for optimal train rescheduling
"""

import pulp
from typing import Dict, List, Tuple, Optional, Set
import logging
import json
import math
import time
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    PLATFORM_ASSIGNMENT = "platform_assignment"
    DEPARTURE_OPTIMIZATION = "departure_optimization" 
    FULL_OPTIMIZATION = "full_optimization"
    EMERGENCY_REROUTING = "emergency_rerouting"
    TRACK_ASSIGNMENT = "track_assignment"
    ORDER_CHANGE = "order_change"

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
    planned_arrival: float  # e_initial_k
    planned_departure: float  # b_initial_k
    min_duration: float  # d_k
    is_planned_stop: bool  # h_k
    origin_direction: int  # o_k (0 or 1)
    
@dataclass
@dataclass
class SegmentData:
    """Segment j in the MILP model - represents a track segment (line or station)"""
    segment_id: str
    name: str
    distance_km: float
    num_tracks: int  # |P_j| - number of tracks on this segment
    is_station: bool  # s_j (1 if station, 0 if line)
    track_lengths: List[float]  # g^track_jt for each track
    meet_separation: float  # DM_j - minimum separation for meeting trains
    follow_separation: float  # DF_j - minimum separation for following trains

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
    computation_time: float

class RailwayMILPOptimizer:
    """
    MILP Optimizer implementing Törnquist & Persson (2007) model
    Event-based formulation for n-tracked railway rescheduling
    """
    
    def __init__(self, trains_data: Dict, stations: Dict, tracks: Dict):
        self.trains_data = trains_data
        self.stations = stations
        self.tracks = self._process_track_data(tracks)
        
        # MILP Model Sets (from paper)
        self.T = set()  # Set of trains
        self.B = set()  # Set of segments
        self.E = set()  # Set of events
        self.K = {}     # K_i: Events for train i (ordered)
        self.L = {}     # L_j: Events on segment j
        self.P = {}     # P_j: Tracks on segment j
        
        # MILP Parameters
        self.M = 1000  # Big-M constant
        self.w = {}    # w_i: Penalty threshold (1 hour = 60 minutes)
        self.c_low = {}    # c^low_i: Low delay cost per minute
        self.c_penalty = {}  # c^penalty_i: Penalty cost for delays > w_i
        self.g_train = {}    # g^train_i: Train length (meters)
        self.g_connection = {}  # g^connection_kk': Connection time
        
        # Decision tracking
        self.pending_decisions = []
        self.decision_history = []
        self.decision_counter = 0
        
        # Optimization parameters
        self.max_optimization_time = 30  # seconds
        self.strategy = 3  # Strategy 3 from paper (limited swaps)
        
        # Initialize MILP sets and parameters
        self._initialize_milp_model()
        
    def _process_track_data(self, tracks: Dict) -> Dict[str, SegmentData]:
        """Process track data into SegmentData objects"""
        processed_tracks = {}
        
        # Create segments for each major section
        segments = [
            ("SBC_Kengeri", "SBC-Kengeri", 25.0, False),
            ("Kengeri_MYA", "Kengeri-MYA", 60.0, False), 
            ("MYA_Channapatna", "MYA-Channapatna", 45.0, False),
            ("Channapatna_MYS", "Channapatna-MYS", 55.0, False),
            ("SBC", "SBC Station", 0.0, True),
            ("Kengeri", "Kengeri Station", 0.0, True),
            ("MYA", "MYA Station", 0.0, True),
            ("Channapatna", "Channapatna Station", 0.0, True),
            ("MYS", "MYS Station", 0.0, True)
        ]
        
        for seg_id, name, distance, is_station in segments:
            # Determine number of tracks based on segment importance
            if seg_id in ["SBC", "MYS"]:
                num_tracks = 4  # Terminal stations
            elif seg_id in ["MYA"]:
                num_tracks = 3  # Major junction
            elif is_station:
                num_tracks = 2  # Other stations
            else:
                num_tracks = 2  # Line segments (double track)
                
            # Track lengths (all 500m for stations, actual length for lines)
            track_lengths = [500.0 if is_station else distance * 1000] * num_tracks
            
            # Separation times based on track type
            meet_sep = 3.0 if is_station else 5.0  # minutes
            follow_sep = 2.0 if is_station else 3.0  # minutes
            
            processed_tracks[seg_id] = SegmentData(
                segment_id=seg_id,
                name=name,
                distance_km=distance,
                num_tracks=num_tracks,
                is_station=is_station,
                track_lengths=track_lengths,
                meet_separation=meet_sep,
                follow_separation=follow_sep
            )
            
        return processed_tracks
    
    def _initialize_milp_model(self):
        """Initialize MILP sets and parameters from train data"""
        current_time = time.time() * 60  # Convert to minutes
        
        # Set T: All trains
        self.T = set(self.trains_data.keys())
        
        # Set B: All segments  
        self.B = set(self.tracks.keys())
        
        # Initialize per-segment track sets P_j
        for segment_id, segment in self.tracks.items():
            self.P[segment_id] = list(range(1, segment.num_tracks + 1))
        
        # Create events E and organize by train (K_i) and segment (L_j)
        self._create_events()
        
        # Initialize cost parameters
        for train_id in self.T:
            train = self.trains_data[train_id]
            priority = train.get('priority', 'medium')
            
            # Cost per minute of delay (paper's c^low_i)
            if priority == 'high':
                self.c_low[train_id] = 10.0
            elif priority == 'medium':
                self.c_low[train_id] = 5.0
            else:  # low priority
                self.c_low[train_id] = 1.0
                
            # Penalty threshold and cost (paper's w_i and c^penalty_i)
            self.w[train_id] = 60.0  # 1 hour threshold
            self.c_penalty[train_id] = 100.0  # High penalty for excessive delays
            
            # Train length (paper's g^train_i)
            self.g_train[train_id] = 500.0  # meters (standard train length)
    
    def _create_events(self):
        """Create events E from train schedules"""
        self.E = set()
        self.K = {train_id: [] for train_id in self.T}
        self.L = {segment_id: [] for segment_id in self.B}
        
        for train_id, train in self.trains_data.items():
            stops = train.get('stops', [])
            if isinstance(stops, str):
                stops = eval(stops)  # Parse string representation of list
                
            dep_time = train.get('dep_time', 0)  # Initial departure time
            current_time = dep_time
            
            # Create events for this train's journey
            train_events = []
            
            for i, station in enumerate(stops):
                # Create station event
                if station in self.tracks:
                    event_id = f"{train_id}_{station}"
                    
                    # Calculate planned times
                    if i == 0:
                        # First station (origin)
                        planned_arrival = dep_time
                        planned_departure = dep_time
                    else:
                        # Calculate travel time from previous station
                        prev_station = stops[i-1]
                        travel_time = self._calculate_travel_time(train_id, prev_station, station)
                        planned_arrival = current_time + travel_time
                        
                        # Add dwell time
                        dwell_time = self._get_dwell_time(train_id, station)
                        planned_departure = planned_arrival + dwell_time
                        
                    current_time = planned_departure
                    
                    # Create TrainEvent
                    event = TrainEvent(
                        event_id=event_id,
                        train_id=train_id,
                        segment_id=station,
                        is_station=True,
                        planned_arrival=planned_arrival,
                        planned_departure=planned_departure,
                        min_duration=self._get_dwell_time(train_id, station),
                        is_planned_stop=True,
                        origin_direction=0 if station == stops[0] else 1
                    )
                    
                    train_events.append(event)
                    self.E.add(event_id)
                    self.L[station].append(event_id)
                    
                # Create line segment event (if not last station)
                if i < len(stops) - 1:
                    next_station = stops[i + 1]
                    line_segment_id = f"{station}_{next_station}"
                    
                    # Check if this line segment exists
                    if line_segment_id not in self.tracks:
                        # Try reverse direction
                        line_segment_id = f"{next_station}_{station}"
                        
                    if line_segment_id in self.tracks:
                        event_id = f"{train_id}_{line_segment_id}"
                        
                        travel_time = self._calculate_travel_time(train_id, station, next_station)
                        
                        event = TrainEvent(
                            event_id=event_id,
                            train_id=train_id,
                            segment_id=line_segment_id,
                            is_station=False,
                            planned_arrival=current_time,
                            planned_departure=current_time + travel_time,
                            min_duration=travel_time,
                            is_planned_stop=False,
                            origin_direction=0
                        )
                        
                        train_events.append(event)
                        self.E.add(event_id)
                        self.L[line_segment_id].append(event_id)
            
            self.K[train_id] = train_events
    
    def _calculate_travel_time(self, train_id: str, from_station: str, to_station: str) -> float:
        """Calculate travel time between stations in minutes"""
        train = self.trains_data[train_id]
        speed_kmh = train.get('speed_kmh', 60)
        
        # Station distances (approximate)
        distances = {
            ('SBC', 'Kengeri'): 25.0,
            ('Kengeri', 'MYA'): 60.0,
            ('MYA', 'Channapatna'): 45.0,
            ('Channapatna', 'MYS'): 55.0
        }
        
        # Get distance (handle both directions)
        distance = distances.get((from_station, to_station)) or distances.get((to_station, from_station), 10.0)
        
        # Calculate time in minutes
        travel_time = (distance / speed_kmh) * 60
        
        return travel_time
    
    def _get_dwell_time(self, train_id: str, station: str) -> float:
        """Get dwell time at station in minutes"""
        train = self.trains_data[train_id]
        train_type = train.get('train_type', 'passenger')
        
        # Base dwell times by station and train type
        dwell_times = {
            'SBC': {'passenger': 5.0, 'freight': 10.0},
            'MYS': {'passenger': 5.0, 'freight': 10.0}, 
            'MYA': {'passenger': 3.0, 'freight': 8.0},
            'Kengeri': {'passenger': 2.0, 'freight': 5.0},
            'Channapatna': {'passenger': 2.0, 'freight': 5.0}
        }
        
        return dwell_times.get(station, {}).get(train_type, 2.0)
    
    def optimize_schedule(self, disrupted_trains: List[str] = None, 
                         optimization_type: str = "total_delay",
                         strategy: int = 3) -> OptimizationResult:
        """
        Main MILP optimization method implementing Törnquist & Persson model
        
        Args:
            disrupted_trains: List of train IDs that are disrupted
            optimization_type: "total_delay" (objective 1a) or "weighted_cost" (objective 1b)
            strategy: Optimization strategy (1-4, default 3 for good balance)
        
        Returns:
            OptimizationResult with optimized schedule and decisions
        """
        start_time = time.time()
        logger.info(f"Starting MILP optimization with strategy {strategy}")
        
        try:
            # Create PuLP problem
            if optimization_type == "total_delay":
                prob = pulp.LpProblem("Railway_Rescheduling_MinDelay", pulp.LpMinimize)
            else:
                prob = pulp.LpProblem("Railway_Rescheduling_WeightedCost", pulp.LpMinimize)
            
            # Decision variables (Table 3 from paper)
            x_begin = {}  # x^begin_k: Start time of event k
            x_end = {}    # x^end_k: End time of event k
            z = {}        # z_k: Delay of event k
            z_final = {}  # z_n_i: Final delay of train i
            x_order = {}  # x_{k,k'}: Binary ordering variable
            y_track = {}  # y_{k,p}: Binary track assignment
            e_penalty = {}  # e_i: Binary penalty activation
            
            # Create decision variables
            self._create_decision_variables(prob, x_begin, x_end, z, z_final, 
                                          x_order, y_track, e_penalty, strategy)
            
            # Add constraints (Sections 4.1-4.3 from paper)
            self._add_sequencing_constraints(prob, x_begin, x_end)
            self._add_duration_constraints(prob, x_begin, x_end)
            self._add_planned_stop_constraints(prob, x_begin)
            self._add_delay_recording_constraints(prob, x_end, z, z_final)
            self._add_track_assignment_constraints(prob, y_track)
            self._add_separation_constraints(prob, x_begin, x_end, x_order, y_track, strategy)
            self._add_train_length_constraints(prob, y_track)
            self._add_penalty_constraints(prob, z_final, e_penalty)
            
            # Set objective function
            if optimization_type == "total_delay":
                # Objective 1a: Minimize total final delay
                prob += pulp.lpSum([z_final[train_id] for train_id in self.T])
            else:
                # Objective 1b: Minimize weighted costs
                prob += pulp.lpSum([
                    self.c_low[train_id] * z_final[train_id] + 
                    self.c_penalty[train_id] * e_penalty[train_id] 
                    for train_id in self.T
                ])
            
            # Solve the problem
            solver = pulp.PULP_CBC_CMD(timeLimit=self.max_optimization_time, msg=False)
            prob.solve(solver)
            
            # Process results
            return self._process_optimization_results(
                prob, x_begin, x_end, z_final, y_track, x_order, 
                start_time, optimization_type, strategy
            )
            
        except Exception as e:
            logger.error(f"MILP optimization failed: {e}")
            return OptimizationResult(
                status="failed",
                objective_value=float('inf'),
                total_delay=0.0,
                delay_reduction=0.0,
                decisions=[],
                optimized_times={},
                track_assignments={},
                order_changes=[],
                computation_time=time.time() - start_time
            )
    
    def _create_decision_variables(self, prob, x_begin, x_end, z, z_final, 
                                 x_order, y_track, e_penalty, strategy):
        """Create MILP decision variables"""
        # Continuous variables for event times
        for event_id in self.E:
            x_begin[event_id] = pulp.LpVariable(f"x_begin_{event_id}", lowBound=0)
            x_end[event_id] = pulp.LpVariable(f"x_end_{event_id}", lowBound=0)
            z[event_id] = pulp.LpVariable(f"z_{event_id}", lowBound=0)
        
        # Final delay variables for each train
        for train_id in self.T:
            z_final[train_id] = pulp.LpVariable(f"z_final_{train_id}", lowBound=0)
            e_penalty[train_id] = pulp.LpVariable(f"e_penalty_{train_id}", cat='Binary')
        
        # Track assignment variables
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            segment_id = event.segment_id
            
            if segment_id in self.P:
                for track in self.P[segment_id]:
                    y_track[(event_id, track)] = pulp.LpVariable(
                        f"y_track_{event_id}_{track}", cat='Binary'
                    )
        
        # Ordering variables (strategy-dependent)
        if strategy >= 2:  # Strategies 2-4 allow reordering
            for segment_id in self.B:
                events_on_segment = self.L[segment_id]
                for i, event1 in enumerate(events_on_segment):
                    for j, event2 in enumerate(events_on_segment):
                        if i != j:
                            x_order[(event1, event2)] = pulp.LpVariable(
                                f"x_order_{event1}_{event2}", cat='Binary'
                            )
    
    def _add_sequencing_constraints(self, prob, x_begin, x_end):
        """Add sequencing constraints (2) from paper"""
        for train_id in self.T:
            events = self.K[train_id]
            for i in range(len(events) - 1):
                current_event = events[i]
                next_event = events[i + 1]
                
                # Constraint (2): x^end_k = x^begin_{k+1}
                prob += x_end[current_event.event_id] == x_begin[next_event.event_id]
    
    def _add_duration_constraints(self, prob, x_begin, x_end):
        """Add duration constraints (3) from paper"""
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            
            # Constraint (3): x^end_k = x^begin_k + d_k
            prob += x_end[event_id] == x_begin[event_id] + event.min_duration
    
    def _add_planned_stop_constraints(self, prob, x_begin):
        """Add planned stop constraints (4) from paper"""
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            
            if event.is_planned_stop:
                # Constraint (4): x^begin_k >= b_initial_k if h_k = 1
                prob += x_begin[event_id] >= event.planned_departure
    
    def _add_delay_recording_constraints(self, prob, x_end, z, z_final):
        """Add delay recording constraints (7) from paper"""
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            
            # Constraint (7): x^end_k - e_initial_k <= z_k
            prob += x_end[event_id] - event.planned_arrival <= z[event_id]
        
        # Final delay is the delay of the last event for each train
        for train_id in self.T:
            events = self.K[train_id]
            if events:
                last_event = events[-1]
                prob += z_final[train_id] >= z[last_event.event_id]
    
    def _add_track_assignment_constraints(self, prob, y_track):
        """Add track assignment constraints (8) from paper"""
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            segment_id = event.segment_id
            
            if segment_id in self.P:
                # Constraint (8): sum y_{k,p} = 1 for p in P_j
                prob += pulp.lpSum([
                    y_track.get((event_id, track), 0) 
                    for track in self.P[segment_id]
                ]) == 1
    
    def _add_separation_constraints(self, prob, x_begin, x_end, x_order, y_track, strategy):
        """Add separation constraints (10-12) from paper"""
        for segment_id in self.B:
            segment = self.tracks[segment_id]
            events_on_segment = self.L[segment_id]
            
            for i, event1_id in enumerate(events_on_segment):
                for j, event2_id in enumerate(events_on_segment):
                    if i != j:
                        event1 = self._get_event_by_id(event1_id)
                        event2 = self._get_event_by_id(event2_id)
                        
                        # Check if events can conflict (same track)
                        for track in self.P[segment_id]:
                            if (event1_id, track) in y_track and (event2_id, track) in y_track:
                                
                                # Get appropriate separation time
                                if event1.origin_direction != event2.origin_direction:
                                    # Meeting trains
                                    sep_time = segment.meet_separation
                                else:
                                    # Following trains
                                    sep_time = segment.follow_separation
                                
                                if strategy >= 2 and (event1_id, event2_id) in x_order:
                                    # Constraints (10a-b, 11a-b) with ordering variables
                                    prob += (x_begin[event2_id] >= x_end[event1_id] + sep_time - 
                                           self.M * (2 - y_track[(event1_id, track)] - 
                                                   y_track[(event2_id, track)] + 
                                                   x_order[(event1_id, event2_id)]))
                                    
                                    prob += (x_begin[event1_id] >= x_end[event2_id] + sep_time - 
                                           self.M * (3 - y_track[(event1_id, track)] - 
                                                   y_track[(event2_id, track)] - 
                                                   x_order[(event1_id, event2_id)]))
                                else:
                                    # Simple separation without reordering
                                    if event1.planned_departure <= event2.planned_departure:
                                        prob += (x_begin[event2_id] >= x_end[event1_id] + sep_time - 
                                               self.M * (2 - y_track[(event1_id, track)] - 
                                                       y_track[(event2_id, track)]))
        
        # Mutual exclusion constraint (12) for ordering variables
        if strategy >= 2:
            for segment_id in self.B:
                events_on_segment = self.L[segment_id]
                for i, event1_id in enumerate(events_on_segment):
                    for j, event2_id in enumerate(events_on_segment):
                        if (event1_id, event2_id) in x_order and (event2_id, event1_id) in x_order:
                            # Constraint (12): κ_kk' + c_kk' <= 1
                            prob += x_order[(event1_id, event2_id)] + x_order[(event2_id, event1_id)] <= 1
    
    def _add_train_length_constraints(self, prob, y_track):
        """Add train length constraints (13) from paper"""
        for event_id in self.E:
            event = self._get_event_by_id(event_id)
            segment_id = event.segment_id
            segment = self.tracks[segment_id]
            train_id = event.train_id
            
            if segment_id in self.P:
                for track_num, track in enumerate(self.P[segment_id]):
                    if (event_id, track) in y_track:
                        track_length = segment.track_lengths[track_num - 1]  # 0-indexed
                        train_length = self.g_train[train_id]
                        
                        # Constraint (13): g^train_i * y_{k,p} <= g^track_jp
                        prob += train_length * y_track[(event_id, track)] <= track_length
    
    def _add_penalty_constraints(self, prob, z_final, e_penalty):
        """Add penalty constraints (14) from paper"""
        for train_id in self.T:
            # Constraint (14): z_n_i - w_i <= M * e_i
            prob += z_final[train_id] - self.w[train_id] <= self.M * e_penalty[train_id]
    
    def _get_event_by_id(self, event_id: str) -> TrainEvent:
        """Get TrainEvent object by event_id"""
        for train_id in self.T:
            for event in self.K[train_id]:
                if event.event_id == event_id:
                    return event
        raise ValueError(f"Event {event_id} not found")
    
    def _process_optimization_results(self, prob, x_begin, x_end, z_final, y_track, 
                                    x_order, start_time, optimization_type, strategy) -> OptimizationResult:
        """Process MILP results and create optimization decisions"""
        computation_time = time.time() - start_time
        
        if prob.status == pulp.LpStatusOptimal:
            status = "optimal"
            objective_value = pulp.value(prob.objective)
            
            # Calculate total delay
            total_delay = sum(pulp.value(z_final[train_id]) or 0 for train_id in self.T)
            
            # Calculate delay reduction (compare with current delays)
            current_total_delay = self._calculate_current_total_delay()
            delay_reduction = max(0, current_total_delay - total_delay)
            
            # Extract optimized times
            optimized_times = {}
            track_assignments = {}
            decisions = []
            order_changes = []
            
            for train_id in self.T:
                optimized_times[train_id] = {
                    'arrival_times': {},
                    'departure_times': {},
                    'delays': {}
                }
                
                for event in self.K[train_id]:
                    event_id = event.event_id
                    
                    if event_id in x_begin:
                        opt_begin = pulp.value(x_begin[event_id]) or event.planned_departure
                        opt_end = pulp.value(x_end[event_id]) or event.planned_arrival
                        
                        optimized_times[train_id]['arrival_times'][event.segment_id] = opt_begin
                        optimized_times[train_id]['departure_times'][event.segment_id] = opt_end
                        
                        # Calculate delay
                        delay = opt_end - event.planned_arrival
                        optimized_times[train_id]['delays'][event.segment_id] = delay
                        
                        # Track assignments
                        for track in self.P.get(event.segment_id, []):
                            if (event_id, track) in y_track:
                                if pulp.value(y_track[(event_id, track)]):
                                    track_assignments[event_id] = track
                        
                        # Create decisions for significant changes
                        if abs(delay) > 1.0:  # More than 1 minute change
                            decision = self._create_optimization_decision(
                                train_id, event, opt_begin, opt_end, delay
                            )
                            decisions.append(decision)
            
            # Extract order changes
            if strategy >= 2:
                for (event1_id, event2_id), var in x_order.items():
                    if pulp.value(var):
                        order_changes.append({
                            'event1': event1_id,
                            'event2': event2_id,
                            'original_order': 'needs_calculation',
                            'new_order': 'event1_before_event2'
                        })
            
            logger.info(f"MILP optimization completed successfully in {computation_time:.2f}s")
            logger.info(f"Total delay reduced from {current_total_delay:.1f} to {total_delay:.1f} minutes")
            
        else:
            status = "infeasible" if prob.status == pulp.LpStatusInfeasible else "failed"
            objective_value = float('inf')
            total_delay = 0.0
            delay_reduction = 0.0
            decisions = []
            optimized_times = {}
            track_assignments = {}
            order_changes = []
            
            logger.warning(f"MILP optimization failed with status: {pulp.LpStatus[prob.status]}")
        
        return OptimizationResult(
            status=status,
            objective_value=objective_value,
            total_delay=total_delay,
            delay_reduction=delay_reduction,
            decisions=decisions,
            optimized_times=optimized_times,
            track_assignments=track_assignments,
            order_changes=order_changes,
            computation_time=computation_time
        )
    
    def _calculate_current_total_delay(self) -> float:
        """Calculate current total delay across all trains"""
        current_time = time.time() * 60  # Convert to minutes
        total_delay = 0.0
        
        for train_id, train in self.trains_data.items():
            current_delay = train.get('delay', 0.0)
            total_delay += current_delay
            
        return total_delay
    
    def _create_optimization_decision(self, train_id: str, event: TrainEvent, 
                                    opt_begin: float, opt_end: float, delay: float) -> OptimizationDecision:
        """Create optimization decision for significant schedule changes"""
        decision_id = self._generate_decision_id()
        
        original_value = {
            'arrival_time': event.planned_arrival,
            'departure_time': event.planned_departure,
            'delay': 0.0
        }
        
        proposed_value = {
            'arrival_time': opt_begin,
            'departure_time': opt_end, 
            'delay': delay
        }
        
        impact = self._analyze_impact(train_id, 'full_optimization', original_value, proposed_value)
        
        train = self.trains_data.get(train_id, {})
        priority = train.get('priority', 'medium')
        
        return OptimizationDecision(
            decision_id=decision_id,
            decision_type=DecisionType.FULL_OPTIMIZATION,
            timestamp=time.time(),
            train_id=train_id,
            station=event.segment_id if event.is_station else None,
            original_value=original_value,
            proposed_value=proposed_value,
            priority=priority,
            impact_analysis=impact,
            status=DecisionStatus.PENDING
        )
    
    def _calculate_duration(self, train_id: str, segment: str) -> float:
        """Calculate travel duration for train on segment"""
        train = self.trains_data[train_id]
        segment_data = self.tracks.get(segment, SegmentData(segment, 10.0, 1, []))
        distance = segment_data.distance_km
        speed = train.get('speed_kmh', 60)
        
        # Add dwell time if stopping at stations
        base_travel_time = (distance / speed) * 60  # minutes
        
        # Add station dwell time based on train type
        dwell_time = 0
        if train.get('train_type') == 'passenger':
            dwell_time = 2  # 2 minutes for passenger trains
        elif train.get('train_type') == 'freight':
            dwell_time = 5  # 5 minutes for freight trains
            
        return base_travel_time + dwell_time
    
    def _analyze_impact(self, train_id: str, optimization_type: str, current_value: Dict, 
                       proposed_value: Dict) -> Dict:
        """Analyze impact of proposed optimization"""
        impact = {
            'delay_change': 0,
            'affected_trains': [],
            'network_impact': 'low',
            'passenger_impact': 'minimal',
            'cost_benefit': 'positive'
        }
        
        train = self.trains_data.get(train_id, {})
        
        # Calculate delay impact
        if 'delay' in current_value and 'delay' in proposed_value:
            impact['delay_change'] = proposed_value['delay'] - current_value['delay']
            
        # Assess network impact based on train priority and delay change
        if train.get('priority') == 'high' and abs(impact['delay_change']) > 5:
            impact['network_impact'] = 'high'
        elif abs(impact['delay_change']) > 10:
            impact['network_impact'] = 'medium'
            
        # Passenger impact for passenger trains
        if train.get('train_type') == 'passenger':
            if impact['delay_change'] > 5:
                impact['passenger_impact'] = 'moderate'
            elif impact['delay_change'] > 10:
                impact['passenger_impact'] = 'significant'
                
        # Cost-benefit analysis
        if impact['delay_change'] < -2:  # Delay reduction
            impact['cost_benefit'] = 'very_positive'
        elif impact['delay_change'] > 5:  # Delay increase
            impact['cost_benefit'] = 'negative'
            
        return impact
    
    def create_platform_assignment_decision(self, train_id: str, station: str, 
                                          current_platform: int, proposed_platform: int,
                                          timestamp: float) -> OptimizationDecision:
        """Create platform assignment decision"""
        train = self.trains_data.get(train_id, {})
        
        current_value = {'platform': current_platform}
        proposed_value = {'platform': proposed_platform}
        
        impact = self._analyze_impact(train_id, 'platform_assignment', current_value, proposed_value)
        
        decision = OptimizationDecision(
            decision_id=self._generate_decision_id(),
            decision_type=DecisionType.PLATFORM_ASSIGNMENT,
            timestamp=timestamp,
            train_id=train_id,
            station=station,
            original_value=current_value,
            proposed_value=proposed_value,
            priority=train.get('priority', 'medium'),
            impact_analysis=impact
        )
        
        self.pending_decisions.append(decision)
        return decision
    
    def create_departure_optimization_decision(self, train_id: str, current_departure: float,
                                             proposed_departure: float, timestamp: float) -> OptimizationDecision:
        """Create departure time optimization decision"""
        train = self.trains_data.get(train_id, {})
        
        current_value = {'departure_time': current_departure}
        proposed_value = {'departure_time': proposed_departure}
        
        impact = self._analyze_impact(train_id, 'departure_optimization', current_value, proposed_value)
        
        decision = OptimizationDecision(
            decision_id=self._generate_decision_id(),
            decision_type=DecisionType.DEPARTURE_OPTIMIZATION,
            timestamp=timestamp,
            train_id=train_id,
            station=None,
            original_value=current_value,
            proposed_value=proposed_value,
            priority=train.get('priority', 'medium'),
            impact_analysis=impact
        )
        
        self.pending_decisions.append(decision)
        return decision
    
    def approve_decision(self, decision_id: str, approval_reason: str = "Approved") -> bool:
        """Approve a pending decision"""
        for decision in self.pending_decisions:
            if decision.decision_id == decision_id:
                decision.status = DecisionStatus.APPROVED
                decision.approval_reason = approval_reason
                logger.info(f"Decision {decision_id} approved: {approval_reason}")
                return True
        return False
    
    def reject_decision(self, decision_id: str, rejection_reason: str = "Rejected") -> bool:
        """Reject a pending decision"""
        for decision in self.pending_decisions:
            if decision.decision_id == decision_id:
                decision.status = DecisionStatus.REJECTED
                decision.approval_reason = rejection_reason
                logger.info(f"Decision {decision_id} rejected: {rejection_reason}")
                return True
        return False
    
    def apply_approved_decisions(self) -> List[OptimizationDecision]:
        """Apply all approved decisions and return list of applied decisions"""
        applied_decisions = []
        
        for decision in self.pending_decisions[:]:
            if decision.status == DecisionStatus.APPROVED:
                if self._apply_decision(decision):
                    decision.status = DecisionStatus.APPLIED
                    applied_decisions.append(decision)
                    self.pending_decisions.remove(decision)
                    self.decision_history.append(decision)
        
        return applied_decisions
    
    def _apply_decision(self, decision: OptimizationDecision) -> bool:
        """Apply a specific decision to the system"""
        try:
            if decision.decision_type == DecisionType.PLATFORM_ASSIGNMENT:
                # Apply platform assignment
                station = decision.station
                platform = decision.proposed_value['platform']
                self.platform_assignments[f"{decision.train_id}_{station}"] = platform
                logger.info(f"Applied platform assignment: Train {decision.train_id} → Platform {platform} at {station}")
                
            elif decision.decision_type == DecisionType.DEPARTURE_OPTIMIZATION:
                # Apply departure time optimization
                train_id = decision.train_id
                new_departure = decision.proposed_value['departure_time']
                if train_id in self.trains_data:
                    old_departure = self.trains_data[train_id]['dep_time']
                    self.trains_data[train_id]['dep_time'] = new_departure
                    logger.info(f"Applied departure optimization: Train {train_id} departure {old_departure} → {new_departure}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply decision {decision.decision_id}: {e}")
            return False
    
    def get_pending_decisions(self) -> List[Dict]:
        """Get all pending decisions"""
        return [decision.to_dict() for decision in self.pending_decisions]
    
    def get_decision_history(self, limit: int = 50) -> List[Dict]:
        """Get recent decision history"""
        return [decision.to_dict() for decision in self.decision_history[-limit:]]
    
    def _build_optimization_model(self, current_time: float, disruption_info: Dict, 
                                current_positions: Dict) -> None:
        """Build the MILP optimization model"""
        self.model = pulp.LpProblem("RailwayOptimization", pulp.LpMinimize)
        self.variables = {}
        
        # Get active trains (not completed)
        active_trains = [train_id for train_id, train in self.trains_data.items()
                        if current_time < train['arr_time'] + train.get('delay', 0)]
        
        if not active_trains:
            return
            
        # Create simplified segment model
        segments = ['departure', 'journey', 'arrival']
        
        # Decision variables
        for train_id in active_trains:
            # Timing variables
            self.variables[f"{train_id}_dep_time"] = pulp.LpVariable(
                f"{train_id}_dep_time", lowBound=current_time)
            self.variables[f"{train_id}_arr_time"] = pulp.LpVariable(
                f"{train_id}_arr_time", lowBound=current_time)
            self.variables[f"{train_id}_delay"] = pulp.LpVariable(
                f"{train_id}_delay", lowBound=0)
            
            # Platform assignment variables
            for station in ['SBC', 'MYA', 'MYS']:
                if station in self.stations:
                    max_platforms = self.stations[station].get('platforms', 2)
                    for platform in range(1, max_platforms + 1):
                        self.variables[f"{train_id}_{station}_platform_{platform}"] = pulp.LpVariable(
                            f"{train_id}_{station}_platform_{platform}", cat='Binary')
        
        # Constraints
        for train_id in active_trains:
            train = self.trains_data[train_id]
            original_dep = train['dep_time'] + train.get('delay', 0)
            original_arr = train['arr_time'] + train.get('delay', 0)
            
            # Journey time constraint
            min_journey_time = (original_arr - original_dep) * 0.9  # At least 90% of original time
            self.model += (self.variables[f"{train_id}_arr_time"] >= 
                          self.variables[f"{train_id}_dep_time"] + min_journey_time)
            
            # Delay calculation
            self.model += (self.variables[f"{train_id}_delay"] >= 
                          self.variables[f"{train_id}_arr_time"] - original_arr)
            
            # Platform assignment constraints
            for station in ['SBC', 'MYA', 'MYS']:
                if station in self.stations:
                    max_platforms = self.stations[station].get('platforms', 2)
                    # Each train gets exactly one platform at each station
                    self.model += pulp.lpSum(
                        self.variables[f"{train_id}_{station}_platform_{p}"]
                        for p in range(1, max_platforms + 1)) == 1
        
        # Headway constraints between trains
        for i, train1 in enumerate(active_trains):
            for j, train2 in enumerate(active_trains):
                if i < j:  # Avoid duplicate constraints
                    # Departure headway
                    self.model += (self.variables[f"{train1}_dep_time"] + self.min_headway <= 
                                  self.variables[f"{train2}_dep_time"] + 
                                  self.M * self.variables.get(f"order_{train1}_{train2}", 0))
                    
                    # Platform conflict constraints
                    for station in ['SBC', 'MYA', 'MYS']:
                        if station in self.stations:
                            max_platforms = self.stations[station].get('platforms', 2)
                            for platform in range(1, max_platforms + 1):
                                # Two trains cannot use same platform simultaneously
                                var1 = self.variables.get(f"{train1}_{station}_platform_{platform}")
                                var2 = self.variables.get(f"{train2}_{station}_platform_{platform}")
                                if var1 and var2:
                                    self.model += var1 + var2 <= 1
        
        # Objective function: minimize weighted delays
        priority_weights = {'high': 10, 'medium': 5, 'low': 1}
        delay_cost = pulp.lpSum(
            priority_weights.get(self.trains_data[train_id].get('priority', 'medium'), 5) *
            self.variables[f"{train_id}_delay"]
            for train_id in active_trains
        )
        
        self.model += delay_cost, "MinimizeWeightedDelay"
    
    def run_live_optimization(self, current_time: float, disruption_info: Dict = None, 
                         current_positions: Dict = None) -> Dict:
        """Run MILP optimization and generate decisions"""
        try:
            start_time = time.time()
            
            # Build optimization model
            self._build_optimization_model(current_time, disruption_info or {}, current_positions or {})
            
            if not hasattr(self, 'model') or not self.variables:
                return { 
                    'success': False,
                    'message': 'No active trains to optimize',
                    'timestamp': current_time
                }
            
            # Solve the model
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self.max_optimization_time)
            status = self.model.solve(solver)
            
            solve_time = time.time() - start_time
            
            if status != pulp.LpStatusOptimal:
                return {
                    'success': False,
                    'message': f'Optimization failed: {pulp.LpStatus[status]}',
                    'solve_time': solve_time
                }
            
            # Generate optimization decisions
            decisions_created = []
            optimized_trains = {}
            
            for train_id in self.trains_data:
                if f"{train_id}_delay" in self.variables:
                    delay_var = self.variables[f"{train_id}_delay"]
                    dep_var = self.variables.get(f"{train_id}_dep_time")
                    
                    if pulp.value(delay_var) is not None:
                        new_delay = pulp.value(delay_var)
                        old_delay = self.trains_data[train_id].get('delay', 0)
                        
                        # Create decisions for significant changes
                        if abs(new_delay - old_delay) > 1:  # More than 1 minute change
                            optimized_trains[train_id] = {
                                'arrival_delay': new_delay,
                                'old_delay': old_delay,
                                'improvement': old_delay - new_delay
                            }
                            
                            if dep_var and pulp.value(dep_var) is not None:
                                new_dep_time = pulp.value(dep_var)
                                old_dep_time = self.trains_data[train_id]['dep_time']
                                
                                if abs(new_dep_time - old_dep_time) > 1:
                                    decision = self.create_departure_optimization_decision(
                                        train_id, old_dep_time, new_dep_time, current_time
                                    )
                                    decisions_created.append(decision.to_dict())
                
                # Check for platform assignment optimizations
                for station in ['SBC', 'MYA', 'MYS']:
                    if station in self.stations:
                        max_platforms = self.stations[station].get('platforms', 2)
                        for platform in range(1, max_platforms + 1):
                            var_name = f"{train_id}_{station}_platform_{platform}"
                            if var_name in self.variables:
                                if pulp.value(self.variables[var_name]) == 1:
                                    # Check if this is different from current assignment
                                    current_assignment = self.platform_assignments.get(
                                        f"{train_id}_{station}", 1)
                                    if platform != current_assignment:
                                        decision = self.create_platform_assignment_decision(
                                            train_id, station, current_assignment, platform, current_time
                                        )
                                        decisions_created.append(decision.to_dict())
            
            # Summary statistics
            total_delay_reduction = sum(
                max(0, data.get('improvement', 0)) for data in optimized_trains.values()
            )
            
            return {
                'success': True,
                'message': f'Optimization completed in {solve_time:.2f}s',
                'trains': optimized_trains,
                'decisions_created': decisions_created,
                'pending_decisions_count': len(self.pending_decisions),
                'optimization_summary': {
                    'total_delay_reduction': total_delay_reduction,
                    'trains_analyzed': len(optimized_trains),
                    'decisions_generated': len(decisions_created),
                    'solve_time': solve_time
                },
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"MILP optimization error: {e}")
            return {
                'success': False,
                'message': f'Optimization error: {str(e)}',
                'timestamp': current_time
            }
    
    def get_optimization_status(self) -> Dict:
        """Get current optimization status"""
        return {
            'pending_decisions': len(self.pending_decisions),
            'decision_history_count': len(self.decision_history),
            'platform_assignments': self.platform_assignments,
            'recent_decisions': self.get_pending_decisions()
        }

    def optimize_station_platform_assignment(self, train_id: str, station: str, 
                                           current_time: float, current_positions: Dict = None) -> Dict:
        """
        Optimize platform assignment for a train arriving at a station.
        Considers current train positions and platform availability.
        
        Args:
            train_id: ID of the arriving train
            station: Station where train is arriving
            current_time: Current simulation time
            current_positions: Current positions of all trains
            
        Returns:
            Dict with optimization result and platform assignment decision
        """
        try:
            start_time = time.time()
            
            if station not in self.stations:
                return {
                    'success': False,
                    'message': f'Station {station} not found',
                    'timestamp': current_time
                }
            
            # Get available platforms at station
            max_platforms = self.stations[station].get('platforms', 2)
            available_platforms = list(range(1, max_platforms + 1))
            
            # Check which platforms are currently occupied
            occupied_platforms = set()
            if current_positions:
                for other_train_id, pos in current_positions.items():
                    if other_train_id != train_id and pos.get('station') == station:
                        # Check if other train is at this station
                        if pos.get('at_station', False):
                            assigned_platform = self.platform_assignments.get(f"{other_train_id}_{station}")
                            if assigned_platform:
                                occupied_platforms.add(assigned_platform)
            
            # Find available platforms
            free_platforms = [p for p in available_platforms if p not in occupied_platforms]
            
            if not free_platforms:
                return {
                    'success': False,
                    'message': f'No platforms available at {station}',
                    'timestamp': current_time
                }
            
            # If only one platform available, assign it directly
            if len(free_platforms) == 1:
                platform = free_platforms[0]
                decision = self.create_platform_assignment_decision(
                    train_id, station, 0, platform, current_time
                )
                return {
                    'success': True,
                    'message': f'Assigned platform {platform} at {station}',
                    'assigned_platform': platform,
                    'decisions_created': [decision.to_dict()],
                    'timestamp': current_time,
                    'computation_time': time.time() - start_time
                }
            
            # Multiple platforms available - use optimization to choose best one
            # Consider factors like:
            # - Platform efficiency (some platforms may be better for certain train types)
            # - Future conflicts (avoid platforms that will be needed soon)
            # - Train priority and type
            
            train = self.trains_data.get(train_id, {})
            train_type = train.get('train_type', 'passenger')
            priority = train.get('priority', 'medium')
            
            # Simple scoring system for platform selection
            platform_scores = {}
            
            for platform in free_platforms:
                score = 0
                
                # Prefer platforms that match train type efficiency
                if train_type == 'passenger' and platform <= 2:  # Main platforms for passengers
                    score += 2
                elif train_type == 'freight' and platform > 2:  # Side platforms for freight
                    score += 2
                
                # Prefer platforms for high priority trains
                if priority == 'high':
                    score += 1
                
                # Check for upcoming conflicts (simplified)
                # In a real system, this would consider scheduled arrivals
                conflict_penalty = 0
                if current_positions:
                    for other_train_id, pos in current_positions.items():
                        if other_train_id != train_id:
                            # Estimate arrival time at station
                            distance_to_station = pos.get('distance_to_station', float('inf'))
                            if distance_to_station < 10:  # Within 10km
                                speed = self.trains_data.get(other_train_id, {}).get('speed_kmh', 60)
                                eta_minutes = (distance_to_station / speed) * 60
                                if eta_minutes < 30:  # Arriving within 30 minutes
                                    conflict_penalty += 1
                
                score -= conflict_penalty
                platform_scores[platform] = score
            
            # Select platform with highest score
            best_platform = max(platform_scores, key=platform_scores.get)
            
            # Create decision for the assignment
            decision = self.create_platform_assignment_decision(
                train_id, station, 0, best_platform, current_time
            )
            
            return {
                'success': True,
                'message': f'Optimized platform {best_platform} assignment at {station}',
                'assigned_platform': best_platform,
                'platform_scores': platform_scores,
                'decisions_created': [decision.to_dict()],
                'timestamp': current_time,
                'computation_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Station platform optimization error: {e}")
            return {
                'success': False,
                'message': f'Optimization error: {str(e)}',
                'timestamp': current_time
            }

    def create_departure_optimization_decision(self, train_id: str, station: str,
                                             current_departure: float, proposed_departure: float,
                                             timestamp: float) -> OptimizationDecision:
        """Create departure time optimization decision"""
        train = self.trains_data.get(train_id, {})
        
        current_value = {
            'departure_time': current_departure,
            'delay': train.get('delay', 0.0)
        }
        
        proposed_value = {
            'departure_time': proposed_departure,
            'delay': proposed_departure - train.get('dep_time', 0)
        }
        
        impact = self._analyze_impact(train_id, 'departure_optimization', current_value, proposed_value)
        
        return OptimizationDecision(
            decision_id=self._generate_decision_id(),
            decision_type=DecisionType.DEPARTURE_OPTIMIZATION,
            timestamp=timestamp,
            train_id=train_id,
            station=station,
            original_value=current_value,
            proposed_value=proposed_value,
            priority=train.get('priority', 'medium'),
            impact_analysis=impact,
            status=DecisionStatus.PENDING
        )

    def apply_disruption(self, train_id: str, delay_minutes: float, station: str = None) -> Dict:
        """Apply a disruption to a specific train"""
        if train_id not in self.trains_data:
            return {"status": "error", "message": f"Train {train_id} not found"}
        
        # Update train delay
        self.trains_data[train_id]['delay'] = self.trains_data[train_id].get('delay', 0) + delay_minutes
        
        # Update departure time if at origin
        if station and station == self.trains_data[train_id].get('stops', [''])[0]:
            current_dep = self.trains_data[train_id].get('dep_time', 0)
            self.trains_data[train_id]['dep_time'] = current_dep + delay_minutes
        
        logger.info(f"Applied {delay_minutes} minute delay to train {train_id}")
        
        return {
            "status": "success",
            "message": f"Applied {delay_minutes} minute delay to train {train_id}",
            "new_delay": self.trains_data[train_id]['delay'],
            "affected_train": train_id
        }

    def get_optimization_summary(self, result: OptimizationResult) -> Dict:
        """Generate a summary of optimization results for display"""
        summary = {
            "optimization_status": result.status,
            "computation_time": round(result.computation_time, 2),
            "objective_value": round(result.objective_value, 2),
            "total_delay_before": round(self._calculate_current_total_delay(), 2),
            "total_delay_after": round(result.total_delay, 2),
            "delay_reduction": round(result.delay_reduction, 2),
            "delay_reduction_percent": 0.0,
            "decisions_count": len(result.decisions),
            "track_assignments_count": len(result.track_assignments),
            "order_changes_count": len(result.order_changes),
            "trains_affected": list(result.optimized_times.keys()),
            "high_impact_decisions": [],
            "recommendations": []
        }
        
        # Calculate percentage improvement
        if summary["total_delay_before"] > 0:
            summary["delay_reduction_percent"] = round(
                (summary["delay_reduction"] / summary["total_delay_before"]) * 100, 1
            )
        
        # Identify high-impact decisions
        for decision in result.decisions:
            impact = decision.impact_analysis.get('delay_change', 0)
            if abs(impact) > 5:  # More than 5 minutes impact
                summary["high_impact_decisions"].append({
                    "train_id": decision.train_id,
                    "type": decision.decision_type.value,
                    "impact": round(impact, 1),
                    "station": decision.station
                })
        
        # Generate recommendations
        if result.status == "optimal":
            if summary["delay_reduction"] > 10:
                summary["recommendations"].append("Significant delay reduction achieved - recommend implementing all decisions")
            elif summary["delay_reduction"] > 2:
                summary["recommendations"].append("Moderate improvements available - consider implementing high-priority decisions")
            else:
                summary["recommendations"].append("Minor improvements - may not be worth operational disruption")
        elif result.status == "infeasible":
            summary["recommendations"].append("No feasible solution found - consider relaxing constraints or extending time horizon")
        else:
            summary["recommendations"].append("Optimization failed - check system constraints and try again")
        
        return summary

    def get_system_status(self) -> Dict:
        """Get overall system status and performance metrics"""
        total_trains = len(self.trains_data)
        delayed_trains = sum(1 for train in self.trains_data.values() if train.get('delay', 0) > 0)
        total_delay = sum(train.get('delay', 0) for train in self.trains_data.values())
        
        avg_delay = total_delay / total_trains if total_trains > 0 else 0
        
        return {
            "total_trains": total_trains,
            "delayed_trains": delayed_trains,
            "on_time_trains": total_trains - delayed_trains,
            "total_delay_minutes": round(total_delay, 2),
            "average_delay": round(avg_delay, 2),
            "pending_decisions": len(self.pending_decisions),
            "approved_decisions": len([d for d in self.decision_history if d.status == DecisionStatus.APPROVED]),
            "rejected_decisions": len([d for d in self.decision_history if d.status == DecisionStatus.REJECTED]),
            "system_efficiency": round((1 - delayed_trains/total_trains) * 100, 1) if total_trains > 0 else 100,
            "segments_count": len(self.tracks),
            "events_count": len(self.E)
        }

# Test function
if __name__ == "__main__":
    # Mock test data
    mock_trains = {
        "12614": {
            "dep_time": 0, "arr_time": 180, "speed_kmh": 65, 
            "stops": ['SBC', 'MYA', 'MYS'], "train_type": "passenger", 
            "priority": "high", "delay": 0
        },
        "16022": {
            "dep_time": 30, "arr_time": 210, "speed_kmh": 45,
            "stops": ['SBC', 'MYS'], "train_type": "freight",
            "priority": "low", "delay": 5
        }
    }
    
    mock_stations = {
        "SBC": {"lon": 77.5946, "lat": 12.9784, "platforms": 6},
        "MYA": {"lon": 76.8977, "lat": 12.523, "platforms": 4}, 
        "MYS": {"lon": 76.6458, "lat": 12.3033, "platforms": 4}
    }
    
    mock_tracks = {
        "track_0": {
            "length_km": 50.0, "capacity": 1, 
            "coordinates": [[77.5946, 12.9784], [76.8977, 12.523]]
        },
        "track_1": {
            "length_km": 110.0, "capacity": 1,
            "coordinates": [[76.8977, 12.523], [76.6458, 12.3033]]
        }
    }
    
    # Test optimization
    optimizer = RailwayMILPOptimizer(mock_trains, mock_stations, mock_tracks)
    result = optimizer.optimize_schedule(0, {}, {})
    
    print("Optimization Result:")
    print(json.dumps(result, indent=2))
    
    print("\nPending Decisions:")
    for decision in optimizer.get_pending_decisions():
        print(f"- {decision['decision_id']}: {decision['decision_type']} for train {decision['train_id']}")
        
    # Test decision approval workflow
    if optimizer.pending_decisions:
        first_decision = optimizer.pending_decisions[0]
        print(f"\nApproving decision {first_decision.decision_id}...")
        optimizer.approve_decision(first_decision.decision_id, "Test approval")
        
        applied = optimizer.apply_approved_decisions()
        print(f"Applied {len(applied)} decisions")