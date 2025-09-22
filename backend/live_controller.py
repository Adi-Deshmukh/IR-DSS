"""
Live Railway Control System
Continuous real-time optimization and track assignment with decision approval workflow
Simulates a real railway control center with live decision making and operator approval
"""

import threading
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from milp_optimizer import RailwayMILPOptimizer, OptimizationDecision, DecisionType, DecisionStatus

logger = logging.getLogger(__name__)

class LiveRailwayController:
    """
    Live railway control system that continuously optimizes train movements
    Handles real-time track assignments and departure scheduling with decision approval workflow
    """
    
    def __init__(self, trains_data: Dict, stations: Dict, tracks: Dict):
        self.trains_data = trains_data
        self.stations = stations
        self.tracks = tracks
        self.optimizer = RailwayMILPOptimizer(trains_data, stations, tracks)
        
        # Live control state
        self.running = False
        self.control_thread = None
        self.optimization_interval = 15  # Optimize every 15 seconds
        self.last_optimization_time = 0
        
        # Decision approval settings
        self.auto_approve_low_impact = True  # Auto-approve decisions with minimal impact
        self.auto_approve_threshold = 2.0    # Auto-approve if delay change < 2 minutes
        
        # Station track management
        self.station_tracks = self._initialize_station_tracks()
        self.track_assignments = {}  # train_id -> assigned_track
        self.departure_queue = {}    # station -> list of trains waiting
        
        # Live events and decisions
        self.live_events = []
        self.optimization_decisions = []
        self.pending_decisions = []
        self.auto_applied_decisions = []
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'decisions_generated': 0,
            'auto_approved': 0,
            'manual_approvals_needed': 0,
            'total_delay_saved': 0.0
        }
        
    def _initialize_station_tracks(self) -> Dict:
        """Initialize platform tracks for each station"""
        station_tracks = {}
        for station_code, station in self.stations.items():
            platforms = station.get('platforms', 2)
            station_tracks[station_code] = {
                'total_platforms': platforms,
                'available_platforms': list(range(1, platforms + 1)),
                'occupied_platforms': {},  # platform_id -> train_id
                'departure_schedule': {}   # platform_id -> departure_time
            }
        return station_tracks
    
    def process_optimization_decisions(self, decisions: List[Dict], current_time: float):
        """Process optimization decisions with approval workflow"""
        for decision_data in decisions:
            decision = OptimizationDecision(**decision_data)
            
            # Check if decision can be auto-approved
            if self._can_auto_approve(decision):
                # Auto-approve and apply immediately
                decision.status = DecisionStatus.APPROVED
                decision.approval_reason = "Auto-approved (low impact)"
                
                if self._apply_live_decision(decision):
                    decision.status = DecisionStatus.APPLIED
                    self.auto_applied_decisions.append(decision)
                    self.optimization_stats['auto_approved'] += 1
                    
                    logger.info(f"Auto-approved and applied: {decision.decision_type.value} for train {decision.train_id}")
                else:
                    logger.error(f"Failed to apply auto-approved decision: {decision.decision_id}")
            else:
                # Requires manual approval
                self.pending_decisions.append(decision)
                self.optimization_stats['manual_approvals_needed'] += 1
                
                logger.info(f"Decision requires approval: {decision.decision_type.value} for train {decision.train_id}")
                
                # Create notification for operators
                self._create_approval_notification(decision)
    
    def _can_auto_approve(self, decision: OptimizationDecision) -> bool:
        """Determine if a decision can be auto-approved"""
        if not self.auto_approve_low_impact:
            return False
            
        impact = decision.impact_analysis
        
        # Auto-approve if delay change is small
        if abs(impact.get('delay_change', 0)) < self.auto_approve_threshold:
            return True
            
        # Auto-approve low-priority trains with minimal network impact
        if (decision.priority == 'low' and 
            impact.get('network_impact') == 'low' and
            impact.get('passenger_impact') == 'minimal'):
            return True
            
        # Auto-approve platform assignments if they improve efficiency
        if (decision.decision_type == DecisionType.PLATFORM_ASSIGNMENT and
            impact.get('cost_benefit') in ['positive', 'very_positive']):
            return True
            
        return False
    
    def approve_decision(self, decision_id: str, approval_reason: str = "Manually approved") -> bool:
        """Approve a pending decision"""
        for decision in self.pending_decisions:
            if decision.decision_id == decision_id:
                decision.status = DecisionStatus.APPROVED
                decision.approval_reason = approval_reason
                
                if self._apply_live_decision(decision):
                    decision.status = DecisionStatus.APPLIED
                    self.pending_decisions.remove(decision)
                    self.optimization_decisions.append(decision)
                    
                    logger.info(f"Manually approved and applied: {decision.decision_id}")
                    return True
                else:
                    logger.error(f"Failed to apply approved decision: {decision.decision_id}")
                    return False
        return False
    
    def reject_decision(self, decision_id: str, rejection_reason: str = "Rejected by operator") -> bool:
        """Reject a pending decision"""
        for decision in self.pending_decisions:
            if decision.decision_id == decision_id:
                decision.status = DecisionStatus.REJECTED
                decision.approval_reason = rejection_reason
                
                self.pending_decisions.remove(decision)
                self.optimization_decisions.append(decision)
                
                logger.info(f"Decision rejected: {decision.decision_id} - {rejection_reason}")
                return True
        return False
    
    def _apply_live_decision(self, decision: OptimizationDecision) -> bool:
        """Apply a decision to the live system"""
        try:
            if decision.decision_type == DecisionType.PLATFORM_ASSIGNMENT:
                # Apply platform assignment
                station = decision.station
                train_id = decision.train_id
                new_platform = decision.proposed_value['platform']
                
                # Update platform assignment
                if station in self.station_tracks:
                    # Release old platform if assigned
                    for platform_id, assigned_train in self.station_tracks[station]['occupied_platforms'].items():
                        if assigned_train == train_id:
                            self.station_tracks[station]['occupied_platforms'].pop(platform_id)
                            self.station_tracks[station]['available_platforms'].append(platform_id)
                            break
                    
                    # Assign new platform
                    self.station_tracks[station]['occupied_platforms'][new_platform] = train_id
                    if new_platform in self.station_tracks[station]['available_platforms']:
                        self.station_tracks[station]['available_platforms'].remove(new_platform)
                    
                    self.track_assignments[f"{train_id}_{station}"] = new_platform
                    
            elif decision.decision_type == DecisionType.DEPARTURE_OPTIMIZATION:
                # Apply departure time optimization
                train_id = decision.train_id
                new_departure = decision.proposed_value['departure_time']
                
                if train_id in self.trains_data:
                    self.trains_data[train_id]['dep_time'] = new_departure
                    
                    # Update departure queue
                    for station_queue in self.departure_queue.values():
                        for i, queued_train in enumerate(station_queue):
                            if queued_train['train_id'] == train_id:
                                station_queue[i]['departure_time'] = new_departure
                                break
            
            # Track performance improvement
            improvement = decision.impact_analysis.get('delay_change', 0)
            if improvement < 0:  # Negative means delay reduction
                self.optimization_stats['total_delay_saved'] += abs(improvement)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply decision {decision.decision_id}: {e}")
            return False
    
    def _create_approval_notification(self, decision: OptimizationDecision):
        """Create notification for operators about pending decision"""
        notification = {
            'type': 'decision_approval_needed',
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'train_id': decision.train_id,
            'station': decision.station,
            'priority': decision.priority,
            'impact': decision.impact_analysis,
            'timestamp': decision.timestamp,
            'message': self._create_decision_message(decision)
        }
        
        self.live_events.append(notification)
    
    def _create_decision_message(self, decision: OptimizationDecision) -> str:
        """Create human-readable message for decision"""
        if decision.decision_type == DecisionType.PLATFORM_ASSIGNMENT:
            old_platform = decision.original_value['platform']
            new_platform = decision.proposed_value['platform']
            return f"Move train {decision.train_id} from Platform {old_platform} to Platform {new_platform} at {decision.station}"
            
        elif decision.decision_type == DecisionType.DEPARTURE_OPTIMIZATION:
            old_time = decision.original_value['departure_time']
            new_time = decision.proposed_value['departure_time']
            delay_change = new_time - old_time
            action = "delay" if delay_change > 0 else "advance"
            return f"{action.title()} train {decision.train_id} departure by {abs(delay_change):.1f} minutes"
            
        return f"Optimization decision for train {decision.train_id}"
    
    def get_pending_decisions(self) -> List[Dict]:
        """Get all pending decisions requiring approval"""
        return [decision.to_dict() for decision in self.pending_decisions]
    
    def get_decision_history(self, limit: int = 20) -> List[Dict]:
        """Get recent decision history"""
        all_decisions = self.optimization_decisions + self.auto_applied_decisions
        return [decision.to_dict() for decision in all_decisions[-limit:]]
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization performance statistics"""
        return self.optimization_stats.copy()
    
    def start_live_control(self):
        """Start the live railway control system"""
        if self.running:
            return
            
        self.running = True
        self.control_thread = threading.Thread(target=self._live_control_loop, daemon=True)
        self.control_thread.start()
        logger.info("Live Railway Control System started")
        
    def stop_live_control(self):
        """Stop the live railway control system"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2)
        logger.info("Live Railway Control System stopped")
        
    def _live_control_loop(self):
        """Main live control loop - runs continuously during simulation"""
        while self.running:
            try:
                # Get current simulation state
                current_time = self._get_current_simulation_time()
                
                # Check for trains entering stations
                station_arrivals = self._detect_station_arrivals(current_time)
                
                # Process each station arrival
                for arrival in station_arrivals:
                    self._handle_station_arrival(arrival, current_time)
                
                # Check for approved platform assignments and assign platforms
                self._process_approved_platform_assignments(current_time)
                
                # Check for trains ready to depart
                departures_ready = self._check_departure_readiness(current_time)
                
                # Optimize departure times and track assignments
                if departures_ready:
                    self._optimize_departures(departures_ready, current_time)
                
                # Continuous system optimization
                if len(self.live_events) > 0:
                    self._run_live_optimization(current_time)
                
                # Wait before next control cycle
                time.sleep(2)  # Control cycle every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in live control loop: {e}")
                time.sleep(5)
    
    def _get_current_simulation_time(self) -> float:
        """Get current simulation time (this will be passed from main app)"""
        # This will be connected to the main simulation time
        return getattr(self, '_current_time', 0)
    
    def update_simulation_time(self, current_time: float):
        """Update the current simulation time from main app"""
        self._current_time = current_time
    
    def _detect_station_arrivals(self, current_time: float) -> List[Dict]:
        """Detect trains arriving at stations"""
        arrivals = []
        
        for train_id, train in self.trains_data.items():
            # Calculate if train is arriving at any station
            position = self._calculate_train_position(train_id, current_time)
            
            if position and position.get('status') == 'arriving':
                station = position.get('current_segment')
                if station and station in self.stations:
                    arrivals.append({
                        'train_id': train_id,
                        'station': station,
                        'arrival_time': current_time,
                        'train_data': train,
                        'position': position
                    })
                    
        return arrivals
    
    def _handle_station_arrival(self, arrival: Dict, current_time: float):
        """Handle a train arriving at a station - request MILP-optimized platform assignment"""
        train_id = arrival['train_id']
        station = arrival['station']
        
        logger.info(f"Live Control: Train {train_id} arriving at {station}, requesting platform assignment")
        
        # Call MILP optimizer for platform assignment
        try:
            result = self.optimizer.optimize_station_platform_assignment(
                train_id, station, current_time, self._position_cache
            )
            
            if result['success']:
                assigned_platform = result['assigned_platform']
                decisions_created = result.get('decisions_created', [])
                
                if decisions_created:
                    # Process the platform assignment decision through approval workflow
                    self.process_optimization_decisions(decisions_created, current_time)
                    
                    # Mark train as waiting for platform approval
                    if not hasattr(self, 'trains_waiting_approval'):
                        self.trains_waiting_approval = {}
                    
                    self.trains_waiting_approval[train_id] = {
                        'station': station,
                        'requested_platform': assigned_platform,
                        'request_time': current_time,
                        'decision_ids': [d['decision_id'] for d in decisions_created]
                    }
                    
                    logger.info(f"Live Control: Train {train_id} requested platform {assigned_platform} "
                               f"at {station}, waiting for approval")
                else:
                    # No decision created - assign directly (fallback)
                    self._assign_platform_directly(train_id, station, assigned_platform, current_time)
            else:
                # Optimization failed - try simple assignment
                logger.warning(f"MILP platform optimization failed: {result['message']}")
                available_platform = self._find_available_platform(station, current_time)
                
                if available_platform:
                    self._assign_platform_directly(train_id, station, available_platform, current_time)
                else:
                    # No platform available - add to departure queue
                    self._queue_train_at_station(train_id, station, current_time, arrival)
                    
        except Exception as e:
            logger.error(f"Error in station arrival handling: {e}")
            # Fallback to simple assignment
            available_platform = self._find_available_platform(station, current_time)
            if available_platform:
                self._assign_platform_directly(train_id, station, available_platform, current_time)
            else:
                self._queue_train_at_station(train_id, station, current_time, arrival)
    
    def _assign_platform_directly(self, train_id: str, station: str, platform_id: int, current_time: float):
        """Assign a platform directly without approval workflow (fallback)"""
        self._assign_platform(train_id, station, platform_id, current_time)
        
        # Calculate and set departure time
        optimized_departure = self._calculate_optimal_departure(
            train_id, station, platform_id, current_time
        )
        self._update_train_departure(train_id, station, optimized_departure)
        
        logger.info(f"Live Control: Train {train_id} directly assigned platform {platform_id} "
                   f"at {station}, departure: {optimized_departure:.1f}")
    
    def _queue_train_at_station(self, train_id: str, station: str, current_time: float, arrival: Dict):
        """Queue a train when no platforms are available"""
        if station not in self.departure_queue:
            self.departure_queue[station] = []
        
        self.departure_queue[station].append({
            'train_id': train_id,
            'arrival_time': current_time,
            'priority': arrival['train_data'].get('priority', 'medium')
        })
        
        logger.info(f"Live Control: Train {train_id} queued at {station} - no platforms available")
    
    def _process_approved_platform_assignments(self, current_time: float):
        """Process approved platform assignment decisions for waiting trains"""
        if not hasattr(self, 'trains_waiting_approval'):
            return
            
        trains_to_remove = []
        
        for train_id, wait_info in self.trains_waiting_approval.items():
            station = wait_info['station']
            requested_platform = wait_info['requested_platform']
            decision_ids = wait_info['decision_ids']
            
            # Check if any of the decisions have been approved
            approved = False
            for decision_id in decision_ids:
                # Check pending decisions
                for decision in self.pending_decisions:
                    if decision.decision_id == decision_id and decision.status == DecisionStatus.APPROVED:
                        approved = True
                        break
                if approved:
                    break
            
            if approved:
                # Assign the platform and set departure time
                self._assign_platform_directly(train_id, station, requested_platform, current_time)
                
                # Remove from waiting list
                trains_to_remove.append(train_id)
                
                logger.info(f"Live Control: Approved platform {requested_platform} assigned to train {train_id} at {station}")
        
        # Remove processed trains
        for train_id in trains_to_remove:
            del self.trains_waiting_approval[train_id]
    
    def _find_available_platform(self, station: str, current_time: float) -> int:
        """Find an available platform at a station"""
        if station not in self.station_tracks:
            return None
            
        station_data = self.station_tracks[station]
        
        # Check each platform
        for platform_id in station_data['available_platforms']:
            if self._is_platform_free(station, platform_id, current_time):
                return platform_id
                
        return None
    
    def _is_platform_free(self, station: str, platform_id: int, current_time: float) -> bool:
        """Check if a platform is free at given time"""
        station_data = self.station_tracks[station]
        
        # Check if platform is currently occupied
        if platform_id in station_data['occupied_platforms']:
            # Check if the occupying train has departed
            occupying_train = station_data['occupied_platforms'][platform_id]
            departure_time = station_data['departure_schedule'].get(platform_id, 0)
            
            if current_time >= departure_time + 2:  # 2 min buffer for departure
                # Platform should be free now
                self._free_platform(station, platform_id)
                return True
            else:
                return False
                
        return True
    
    def _assign_platform(self, train_id: str, station: str, platform_id: int, current_time: float):
        """Assign a train to a platform"""
        station_data = self.station_tracks[station]
        station_data['occupied_platforms'][platform_id] = train_id
        self.track_assignments[train_id] = f"{station}-Platform-{platform_id}"
    
    def _free_platform(self, station: str, platform_id: int):
        """Free up a platform"""
        station_data = self.station_tracks[station]
        if platform_id in station_data['occupied_platforms']:
            train_id = station_data['occupied_platforms'][platform_id]
            del station_data['occupied_platforms'][platform_id]
            
            if platform_id in station_data['departure_schedule']:
                del station_data['departure_schedule'][platform_id]
    
    def _calculate_optimal_departure(self, train_id: str, station: str, 
                                   platform_id: int, arrival_time: float) -> float:
        """Calculate optimal departure time considering other trains"""
        train = self.trains_data[train_id]
        
        # Base dwell time
        base_dwell = self._get_base_dwell_time(station, train['train_type'], train['priority'])
        earliest_departure = arrival_time + base_dwell
        
        # Check conflicts with other trains
        optimal_departure = self._find_conflict_free_departure(
            station, platform_id, earliest_departure, train
        )
        
        return optimal_departure
    
    def _find_conflict_free_departure(self, station: str, platform_id: int, 
                                    earliest_departure: float, train: Dict) -> float:
        """Find departure time that doesn't conflict with other trains"""
        station_data = self.station_tracks[station]
        
        # Check departures from all platforms at this station
        existing_departures = []
        for pid, dep_time in station_data['departure_schedule'].items():
            if pid != platform_id:  # Don't check same platform
                existing_departures.append(dep_time)
        
        # Sort existing departures
        existing_departures.sort()
        
        # Find a slot with minimum 5-minute headway
        min_headway = 5  # 5 minutes minimum between departures
        optimal_time = earliest_departure
        
        for existing_dep in existing_departures:
            if abs(optimal_time - existing_dep) < min_headway:
                # Adjust departure time
                if train['priority'] == 'high':
                    # High priority trains can depart slightly earlier
                    optimal_time = existing_dep - min_headway
                else:
                    # Other trains wait
                    optimal_time = existing_dep + min_headway
        
        # Ensure we don't go earlier than earliest possible
        optimal_time = max(optimal_time, earliest_departure)
        
        return optimal_time
    
    def _get_base_dwell_time(self, station: str, train_type: str, priority: str) -> float:
        """Get base dwell time for a train at a station"""
        base_times = {
            'SBC': {'passenger': 3, 'freight': 8},
            'MYA': {'passenger': 2, 'freight': 5},
            'MYS': {'passenger': 3, 'freight': 8},
            'Kengeri': {'passenger': 1, 'freight': 3},
            'Bangarpet': {'passenger': 1, 'freight': 4},
            'Channapatna': {'passenger': 1, 'freight': 3}
        }
        
        base_time = base_times.get(station, {}).get(train_type, 2)
        
        # Adjust based on priority
        if priority == 'high':
            return base_time * 0.8
        elif priority == 'low':
            return base_time * 1.2
        else:
            return base_time
    
    def _update_train_departure(self, train_id: str, station: str, departure_time: float):
        """Update train's departure time in the system"""
        if train_id in self.trains_data:
            # Find the station in the train's route and update timing
            train = self.trains_data[train_id]
            
            # This would update the train's schedule in the main system
            # For now, we'll store it locally and apply during optimization
            if not hasattr(train, 'live_schedule_updates'):
                train['live_schedule_updates'] = {}
                
            train['live_schedule_updates'][station] = {
                'departure_time': departure_time,
                'updated_at': time.time()
            }
    
    def _check_departure_readiness(self, current_time: float) -> List[Dict]:
        """Check which trains are ready to depart"""
        ready_departures = []
        
        for station, station_data in self.station_tracks.items():
            for platform_id, train_id in station_data['occupied_platforms'].items():
                scheduled_departure = station_data['departure_schedule'].get(platform_id, 0)
                
                if current_time >= scheduled_departure - 1:  # 1 min before departure
                    ready_departures.append({
                        'train_id': train_id,
                        'station': station,
                        'platform': platform_id,
                        'scheduled_departure': scheduled_departure
                    })
        
        return ready_departures
    
    def _optimize_departures(self, departures: List[Dict], current_time: float):
        """Optimize departure times for trains ready to leave"""
        if not departures:
            return
            
        # Quick local optimization for departure times
        for departure in departures:
            train_id = departure['train_id']
            station = departure['station']
            platform_id = departure['platform']
            
            # Check if we can optimize this departure
            optimized_time = self._micro_optimize_departure(departure, current_time)
            
            if optimized_time != departure['scheduled_departure']:
                # Update the departure time
                self.station_tracks[station]['departure_schedule'][platform_id] = optimized_time
                
                decision = {
                    'timestamp': current_time,
                    'type': 'departure_optimization',
                    'train_id': train_id,
                    'station': station,
                    'platform': platform_id,
                    'original_time': departure['scheduled_departure'],
                    'optimized_time': optimized_time,
                    'time_saved': departure['scheduled_departure'] - optimized_time
                }
                
                self.optimization_decisions.append(decision)
                logger.info(f"Live Control: Optimized departure of {train_id} from {station} "
                           f"by {departure['scheduled_departure'] - optimized_time:.1f} minutes")
    
    def _micro_optimize_departure(self, departure: Dict, current_time: float) -> float:
        """Perform micro-optimization for a single departure"""
        train_id = departure['train_id']
        station = departure['station']
        scheduled = departure['scheduled_departure']
        
        # Can we depart earlier without conflicts?
        min_departure = current_time + 0.5  # At least 30 seconds from now
        earliest_safe = max(min_departure, scheduled - 3)  # Max 3 min earlier
        
        # Check for conflicts with other trains
        safe_departure = self._find_conflict_free_departure(
            station, departure['platform'], earliest_safe, self.trains_data[train_id]
        )
        
        return min(safe_departure, scheduled)  # Don't delay beyond original schedule
    
    def _run_live_optimization(self, current_time: float):
        """Run full MILP optimization with decision approval workflow"""
        # Only run optimization if enough time has passed
        if current_time - self.last_optimization_time < self.optimization_interval:
            return
            
        try:
            # Get current train positions
            current_positions = {}
            for train_id in self.trains_data.keys():
                pos = self._calculate_train_position(train_id, current_time)
                if pos:
                    current_positions[train_id] = pos
            
            # Run MILP optimization
            result = self.optimizer.run_live_optimization(
                current_time, 
                None,  # No specific disruption
                current_positions
            )
            
            if result['success']:
                self.optimization_stats['total_optimizations'] += 1
                
                # Process generated decisions with approval workflow
                decisions_created = result.get('decisions_created', [])
                if decisions_created:
                    self.optimization_stats['decisions_generated'] += len(decisions_created)
                    self.process_optimization_decisions(decisions_created, current_time)
                    
                    logger.info(f"Live optimization generated {len(decisions_created)} decisions at time {current_time}")
                
                # Update last optimization time
                self.last_optimization_time = current_time
                
                # Add optimization event to live events
                self.live_events.append({
                    'type': 'optimization_completed',
                    'timestamp': current_time,
                    'decisions_generated': len(decisions_created),
                    'auto_approved': len([d for d in decisions_created if self._can_auto_approve_dict(d)]),
                    'manual_approval_needed': len([d for d in decisions_created if not self._can_auto_approve_dict(d)]),
                    'message': f"Live optimization completed - {len(decisions_created)} decisions generated"
                })
                
            else:
                logger.warning(f"Live optimization failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error in live optimization: {e}")
    
    def _can_auto_approve_dict(self, decision_dict: Dict) -> bool:
        """Check if a decision dictionary can be auto-approved"""
        if not self.auto_approve_low_impact:
            return False
            
        impact = decision_dict.get('impact_analysis', {})
        priority = decision_dict.get('priority', 'medium')
        decision_type = decision_dict.get('decision_type', '')
        
        # Auto-approve if delay change is small
        if abs(impact.get('delay_change', 0)) < self.auto_approve_threshold:
            return True
            
        # Auto-approve low-priority trains with minimal network impact
        if (priority == 'low' and 
            impact.get('network_impact') == 'low' and
            impact.get('passenger_impact') == 'minimal'):
            return True
            
        # Auto-approve platform assignments if they improve efficiency
        if (decision_type == 'platform_assignment' and
            impact.get('cost_benefit') in ['positive', 'very_positive']):
            return True
            
        return False
    
    def _apply_optimization_results(self, result: Dict, current_time: float):
        """Apply MILP optimization results to the live system"""
        optimized_trains = result.get('trains', {})
        
        for train_id, optimization in optimized_trains.items():
            if train_id in self.trains_data:
                # Apply optimized delays/timings
                new_delay = optimization.get('arrival_delay', 0)
                old_delay = self.trains_data[train_id].get('delay', 0)
                
                if abs(new_delay - old_delay) > 0.5:
                    self.trains_data[train_id]['delay'] = new_delay
                    
                    decision = {
                        'timestamp': current_time,
                        'type': 'full_optimization',
                        'train_id': train_id,
                        'delay_change': new_delay - old_delay,
                        'reason': 'MILP system optimization'
                    }
                    
                    self.optimization_decisions.append(decision)
    
    def _calculate_train_position(self, train_id: str, current_time: float) -> Dict:
        """Calculate current train position (uses cached positions from main app)"""
        return getattr(self, '_position_cache', {}).get(train_id, None)
    
    def set_position_calculator(self, calc_func):
        """Set the position calculation function from main app"""
        self._external_calc_position = calc_func
    
    def update_train_positions(self, positions: Dict):
        """Update train positions from main simulation"""
        self._position_cache = positions
    
    def get_live_decisions(self, limit: int = 10) -> List[Dict]:
        """Get recent live optimization decisions"""
        return self.optimization_decisions[-limit:] if self.optimization_decisions else []
    
    def get_station_status(self) -> Dict:
        """Get current status of all stations and platforms"""
        status = {}
        for station, data in self.station_tracks.items():
            status[station] = {
                'total_platforms': data['total_platforms'],
                'occupied_platforms': len(data['occupied_platforms']),
                'available_platforms': len(data['available_platforms']) - len(data['occupied_platforms']),
                'current_occupancy': dict(data['occupied_platforms']),
                'departure_schedule': dict(data['departure_schedule'])
            }
        return status