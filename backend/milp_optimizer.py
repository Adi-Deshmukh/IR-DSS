"""
Enhanced MILP Optimizer for Railway Decision Support System
Based on Törnquist & Persson (2007) "N-tracked Railway Traffic Re-scheduling During Disturbances"
Implements event-based MILP formulation for optimal train rescheduling
"""

import pulp
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Enums and Data Classes ---

class DecisionType(Enum):
    FULL_OPTIMIZATION = "full_optimization"
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
    priority: str
    impact_analysis: Dict[str, Any]
    details: Dict[str, Any]
    station: Optional[str] = None
    status: DecisionStatus = DecisionStatus.PENDING
    approval_reason: Optional[str] = None
    
    def to_dict(self):
        data = asdict(self)
        data['decision_type'] = self.decision_type.value
        data['status'] = self.status.value
        return data

@dataclass
class TrainEvent:
    event_id: str
    train_id: str
    segment_id: str
    is_station: bool
    planned_arrival: float
    planned_departure: float
    min_duration: float

@dataclass
class SegmentData:
    segment_id: str
    name: str
    num_tracks: int
    is_station: bool
    length_km: float = 0.0

@dataclass
class OptimizationResult:
    status: str
    objective_value: float
    total_delay: float
    delay_reduction: float
    decisions: List[OptimizationDecision]
    computation_time: float
    optimized_times: Dict = field(default_factory=dict)
    track_assignments: Dict = field(default_factory=dict)
    order_changes: List = field(default_factory=list)

# --- Main Optimizer Class ---

class RailwayMILPOptimizer:
    def __init__(self, trains_data: Dict, stations: Dict, tracks: Dict):
        self.initial_trains_data = {k: v.copy() for k, v in trains_data.items()}
        self.stations = stations
        self.tracks = self._process_track_data(tracks)
        
        self.M = 1000  # Big-M constant
        self.max_optimization_time = 30  # seconds

        self.pending_decisions: Dict[str, OptimizationDecision] = {}
        self.decision_history: List[OptimizationDecision] = []
        
        self._initialize_milp_model_data()

    def reset(self):
        """Resets the optimizer to its initial state."""
        self.pending_decisions.clear()
        self.decision_history.clear()
        self._initialize_milp_model_data()
        logger.info("MILP Optimizer has been reset.")

    def _process_track_data(self, tracks: Dict) -> Dict[str, SegmentData]:
        processed = {}
        # Process line segments
        for track_id, track in tracks.items():
            processed[track_id] = SegmentData(
                segment_id=track_id,
                name=track['name'],
                num_tracks=1, # Assume single track segments for simplicity
                is_station=False,
                length_km=track['length_km']
            )
        # Process station segments
        for station_code, station in self.stations.items():
            processed[station_code] = SegmentData(
                segment_id=station_code,
                name=station['name'],
                num_tracks=station.get('platforms', 2),
                is_station=True
            )
        return processed

    def _initialize_milp_model_data(self):
        self.T = set(self.initial_trains_data.keys())
        self.B = set(self.tracks.keys())
        self._create_events()
        
    def _create_events(self):
        self.E: Dict[str, TrainEvent] = {}
        self.K: Dict[str, List[str]] = {train_id: [] for train_id in self.T}
        self.L: Dict[str, List[str]] = {segment_id: [] for segment_id in self.B}

        for train_id, train in self.initial_trains_data.items():
            for i, stop in enumerate(train['stops']):
                # Create station event
                event_id = f"{train_id}_{stop}"
                is_station = self.tracks[stop].is_station

                # Simplified time calculation
                journey_fraction = i / (len(train['stops']) - 1) if len(train['stops']) > 1 else 0
                total_duration = train['base_arr_time'] - train['base_dep_time']
                
                planned_arrival = train['base_dep_time'] + (total_duration * journey_fraction)
                min_duration = 2.0 if is_station else 0 # 2 min dwell time
                planned_departure = planned_arrival + min_duration

                event = TrainEvent(event_id, train_id, stop, is_station, planned_arrival, planned_departure, min_duration)
                self.E[event_id] = event
                self.K[train_id].append(event_id)
                self.L[stop].append(event_id)

    def optimize_schedule(self, disrupted_trains: List[str] = None, strategy: int = 3) -> OptimizationResult:
        start_time = time.time()
        
        prob = pulp.LpProblem("RailwayRescheduling", pulp.LpMinimize)
        
        # --- Variables ---
        x_begin = {k: pulp.LpVariable(f"x_begin_{k}", 0) for k in self.E}
        x_end = {k: pulp.LpVariable(f"x_end_{k}", 0) for k in self.E}
        z_final = {i: pulp.LpVariable(f"z_final_{i}", 0) for i in self.T}
        y_track = {
            (k, p): pulp.LpVariable(f"y_{k}_{p}", cat='Binary')
            for k, e in self.E.items() for p in range(self.tracks[e.segment_id].num_tracks)
        }

        # --- Objective Function: Minimize total final delay ---
        prob += pulp.lpSum(z_final.values()), "TotalDelay"

        # --- Constraints ---
        for event_id, event in self.E.items():
            # Duration constraint
            prob += x_end[event_id] >= x_begin[event_id] + event.min_duration

            # Track assignment constraint
            prob += pulp.lpSum(y_track[(event_id, p)] for p in range(self.tracks[event.segment_id].num_tracks)) == 1

        for train_id in self.T:
            # Sequencing constraint
            train_events = self.K[train_id]
            for i in range(len(train_events) - 1):
                prob += x_begin[train_events[i+1]] >= x_end[train_events[i]]

            # Final delay calculation
            if train_events:
                last_event_id = train_events[-1]
                last_event = self.E[last_event_id]
                prob += z_final[train_id] >= x_end[last_event_id] - last_event.planned_departure
        
        # Separation constraints (simplified)
        for segment_id in self.B:
            events_on_segment = self.L[segment_id]
            headway = 5.0 # 5 minute headway
            for i, k1_id in enumerate(events_on_segment):
                for j, k2_id in enumerate(events_on_segment):
                    if i < j:
                        for p in range(self.tracks[segment_id].num_tracks):
                            # If two events are on the same track, they must be separated in time
                            prob += x_begin[k2_id] >= x_end[k1_id] + headway - self.M * (3 - y_track[k1_id, p] - y_track[k2_id, p] - 1)


        # Apply disruptions
        if disrupted_trains:
            for train_id in disrupted_trains:
                if train_id in self.initial_trains_data:
                    delay = self.initial_trains_data[train_id].get('delay', 0)
                    first_event_id = self.K[train_id][0]
                    first_event = self.E[first_event_id]
                    prob += x_begin[first_event_id] >= first_event.planned_arrival + delay

        # --- Solve ---
        solver = pulp.PULP_CBC_CMD(timeLimit=self.max_optimization_time, msg=False)
        prob.solve(solver)
        
        return self._process_results(prob, x_begin, x_end, z_final, start_time)

    def _process_results(self, prob, x_begin, x_end, z_final, start_time) -> OptimizationResult:
        computation_time = time.time() - start_time
        if prob.status != pulp.LpStatusOptimal:
            return OptimizationResult(pulp.LpStatus[prob.status], 0, 0, 0, [], computation_time)

        objective_value = pulp.value(prob.objective)
        
        # Calculate delay reduction
        current_total_delay = sum(t.get('delay', 0) for t in self.initial_trains_data.values())
        optimized_total_delay = sum(pulp.value(z) for z in z_final.values())
        delay_reduction = current_total_delay - optimized_total_delay

        decisions = []
        for train_id in self.T:
            new_delay = pulp.value(z_final[train_id])
            old_delay = self.initial_trains_data[train_id].get('delay', 0)
            
            # Create a decision if there's a significant change
            if abs(new_delay - old_delay) > 1.0:
                decision = self._create_decision(train_id, old_delay, new_delay)
                decisions.append(decision)
                self.pending_decisions[decision.decision_id] = decision
        
        return OptimizationResult(
            status="optimal",
            objective_value=objective_value,
            total_delay=optimized_total_delay,
            delay_reduction=delay_reduction,
            decisions=decisions,
            computation_time=computation_time
        )
    
    def _create_decision(self, train_id, old_delay, new_delay) -> OptimizationDecision:
        train = self.initial_trains_data[train_id]
        delay_change = new_delay - old_delay
        
        return OptimizationDecision(
            decision_id=str(uuid.uuid4()),
            decision_type=DecisionType.FULL_OPTIMIZATION,
            timestamp=time.time(),
            train_id=train_id,
            priority=train['priority'],
            impact_analysis={'delay_change': delay_change},
            details={'old_delay': old_delay, 'new_delay': new_delay}
        )
    
    def apply_disruption(self, train_id: str, delay_minutes: float):
        """Applies a disruption to the base data for the next optimization run."""
        if train_id in self.initial_trains_data:
            self.initial_trains_data[train_id]['delay'] = self.initial_trains_data[train_id].get('delay', 0) + delay_minutes
            logger.info(f"Disruption recorded for train {train_id}. New total delay: {self.initial_trains_data[train_id]['delay']} mins.")
        else:
            logger.warning(f"Attempted to apply disruption to non-existent train {train_id}")
    
    def get_pending_decisions(self) -> List[Dict]:
        return [d.to_dict() for d in self.pending_decisions.values()]

    def approve_decision(self, decision_id: str) -> Optional[OptimizationDecision]:
        if decision_id in self.pending_decisions:
            decision = self.pending_decisions.pop(decision_id)
            decision.status = DecisionStatus.APPROVED
            self.decision_history.append(decision)
            return decision
        return None

    def apply_decision(self, decision: OptimizationDecision, live_trains_data: Dict) -> bool:
        """Applies a decision to the live simulation data."""
        try:
            train_id = decision.train_id
            if train_id in live_trains_data:
                if decision.decision_type == DecisionType.FULL_OPTIMIZATION:
                    new_delay = decision.details.get('new_delay')
                    if new_delay is not None:
                        # Update the live train data
                        live_trains_data[train_id]['delay'] = new_delay
                        # Also update the base data for future optimizations
                        if train_id in self.initial_trains_data:
                            self.initial_trains_data[train_id]['delay'] = new_delay
                        logger.info(f"Applied new delay of {new_delay:.2f} mins to train {train_id}.")
                        return True
            return False
        except Exception as e:
            logger.error(f"Error applying decision {decision.decision_id}: {e}")
            return False

    def get_optimization_summary(self, result: OptimizationResult) -> Dict:
        return {
            "optimization_status": result.status,
            "computation_time": round(result.computation_time, 2),
            "objective_value": round(result.objective_value, 2),
            "total_delay_before": round(sum(t.get('delay', 0) for t in self.initial_trains_data.values()), 2),
            "total_delay_after": round(result.total_delay, 2),
            "delay_reduction": round(result.delay_reduction, 2),
            "decisions_generated": len(result.decisions)
        }

    