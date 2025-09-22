#!/usr/bin/env python3
"""
Demo script for MILP Railway Decision Support System
Demonstrates the complete optimization workflow with visual feedback
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:5000"

def print_header(title):
    print("\n" + "="*60)
    print(f"🚂 {title}")
    print("="*60)

def print_step(step, description):
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def make_request(endpoint, method="GET", data=None):
    """Make API request with error handling"""
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

def show_system_status():
    """Display current system status"""
    result = make_request("/optimization_summary")
    if result and result.get('success'):
        summary = result['summary']
        status = summary['overall_status']
        color = summary['status_color']
        
        # Color mapping for terminal
        colors = {
            'green': '\033[92m',
            'yellow': '\033[93m', 
            'orange': '\033[91m',
            'red': '\033[91m'
        }
        reset = '\033[0m'
        
        print(f"📊 System Status: {colors.get(color, '')}{status}{reset}")
        print(f"⚡ Efficiency: {summary['efficiency_percent']}%")
        print(f"🚄 Total Trains: {summary['total_trains']}")
        print(f"✅ On Time: {summary['on_time_trains']}")
        print(f"⏰ Delayed: {summary['delayed_trains']}")
        print(f"⏱️ Avg Delay: {summary['average_delay_minutes']:.1f} min")
        
        if summary.get('pending_decisions', 0) > 0:
            print(f"⚠️ Pending Decisions: {summary['pending_decisions']}")

def demonstrate_disruption():
    """Demonstrate applying a disruption"""
    print_step(1, "Apply Train Disruption")
    
    # Apply disruption to train 12614 (high priority passenger)
    disruption_data = {
        "train_id": "12614",
        "delay_minutes": 20,
        "reason": "Signal failure at junction"
    }
    
    print(f"🚨 Applying 20-minute delay to Train 12614 (High Priority Passenger)")
    result = make_request("/disrupt", "POST", disruption_data)
    
    if result and result.get('success'):
        print(f"✅ Disruption applied successfully")
        print(f"📍 Affected train: {result.get('train_affected', 'N/A')}")
        
        # Show updated status
        print("\n📊 Updated System Status:")
        show_system_status()
    else:
        print("❌ Failed to apply disruption")
        return False
    
    return True

def run_optimization():
    """Run MILP optimization"""
    print_step(2, "Run MILP Optimization")
    
    optimization_data = {
        "delayed_train": "12614",
        "delay_minutes": 20,
        "optimization_type": "total_delay",
        "strategy": 3
    }
    
    print("🧮 Running MILP optimization (Törnquist & Persson model)...")
    print("   Objective: Minimize total delay")
    print("   Strategy: 3 (Limited swaps for scalability)")
    
    start_time = time.time()
    result = make_request("/optimize", "POST", optimization_data)
    end_time = time.time()
    
    if result and result.get('success'):
        print(f"✅ Optimization completed in {end_time - start_time:.2f}s")
        print(f"🎯 Status: {result.get('status', 'unknown')}")
        print(f"⏱️ Computation time: {result.get('computation_time', 0):.2f}s")
        
        summary = result.get('summary', {})
        if summary:
            print(f"📈 Objective value: {summary.get('objective_value', 0):.2f}")
            print(f"⏰ Total delay before: {summary.get('total_delay_before', 0):.1f} min")
            print(f"⏰ Total delay after: {summary.get('total_delay_after', 0):.1f} min")
            print(f"📉 Delay reduction: {summary.get('delay_reduction', 0):.1f} min")
            
            if summary.get('delay_reduction_percent', 0) > 0:
                print(f"📊 Improvement: {summary['delay_reduction_percent']:.1f}%")
        
        decisions = result.get('decisions', {})
        auto_applied = decisions.get('auto_applied', [])
        pending = decisions.get('pending_approval', [])
        
        print(f"\n🤖 Auto-applied decisions: {len(auto_applied)}")
        for decision in auto_applied[:3]:  # Show first 3
            print(f"   • {decision['decision_type']} for Train {decision['train_id']}")
            impact = decision.get('impact_analysis', {}).get('delay_change', 0)
            print(f"     Impact: {impact:.1f} min delay change")
        
        print(f"\n👨‍💼 Decisions requiring admin approval: {len(pending)}")
        for decision in pending[:3]:  # Show first 3
            print(f"   • {decision['decision_type']} for Train {decision['train_id']}")
            impact = decision.get('impact_analysis', {}).get('delay_change', 0)
            print(f"     Impact: {impact:.1f} min delay change")
            print(f"     Priority: {decision['priority']}")
        
        return result
    else:
        print("❌ Optimization failed")
        return None

def show_pending_decisions():
    """Show pending decisions requiring approval"""
    print_step(3, "Review Pending Decisions")
    
    result = make_request("/pending_decisions")
    if result and result.get('success'):
        decisions = result.get('decisions', [])
        print(f"📋 Found {len(decisions)} decisions requiring approval")
        
        for i, decision in enumerate(decisions, 1):
            print(f"\n🔍 Decision {i}: {decision['decision_id']}")
            print(f"   Type: {decision['decision_type']}")
            print(f"   Train: {decision['train_id']} ({decision.get('priority', 'medium')} priority)")
            
            if decision.get('station'):
                print(f"   Station: {decision['station']}")
            
            impact = decision.get('impact_analysis', {})
            delay_change = impact.get('delay_change', 0)
            network_impact = impact.get('network_impact', 'unknown')
            
            print(f"   Delay impact: {delay_change:.1f} minutes")
            print(f"   Network impact: {network_impact}")
            
            # Show original vs proposed values
            original = decision.get('original_value', {})
            proposed = decision.get('proposed_value', {})
            
            if 'departure_time' in original and 'departure_time' in proposed:
                print(f"   Departure: {original['departure_time']:.1f} → {proposed['departure_time']:.1f} min")
            
            if 'delay' in proposed:
                print(f"   New delay: {proposed['delay']:.1f} minutes")
        
        return decisions
    else:
        print("❌ Failed to get pending decisions")
        return []

def approve_decision(decision_id, train_id):
    """Approve a specific decision"""
    approval_data = {
        "decision_id": decision_id,
        "reason": f"Approved for demo - optimizes Train {train_id} schedule"
    }
    
    result = make_request("/approve_decision", "POST", approval_data)
    if result and result.get('success'):
        print(f"✅ Decision {decision_id} approved and applied")
        print(f"🚄 Train {train_id} schedule updated")
        return True
    else:
        print(f"❌ Failed to approve decision {decision_id}")
        return False

def show_optimization_results():
    """Show detailed optimization results"""
    print_step(4, "Optimization Results Summary")
    
    result = make_request("/optimization_results")
    if result and result.get('success'):
        results = result['results']
        
        # System status
        system_status = results.get('system_status', {})
        print(f"🎯 System Efficiency: {system_status.get('system_efficiency', 0):.1f}%")
        print(f"⏰ Average Delay: {system_status.get('average_delay', 0):.1f} minutes")
        print(f"🚄 Total Trains: {system_status.get('total_trains', 0)}")
        print(f"⏱️ Delayed Trains: {system_status.get('delayed_trains', 0)}")
        
        # Optimization impact
        impact = results.get('optimization_impact', {})
        print(f"\n📈 Optimization Impact (Last Hour):")
        print(f"   Delay reduction: {impact.get('total_delay_reduction_1h', 0):.1f} minutes")
        print(f"   Decisions approved: {impact.get('decisions_approved_1h', 0)}")
        print(f"   Decisions rejected: {impact.get('decisions_rejected_1h', 0)}")
        
        # Most delayed trains
        train_perf = results.get('train_performance', {})
        delayed_trains = train_perf.get('most_delayed_trains', [])
        if delayed_trains:
            print(f"\n🚨 Most Delayed Trains:")
            for train in delayed_trains[:3]:
                print(f"   • Train {train['train_id']}: {train['delay']:.1f} min ({train['train_type']})")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 System Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
    
    # Final status
    print(f"\n📊 Final System Status:")
    show_system_status()

def main():
    """Main demonstration workflow"""
    print_header("MILP Railway Decision Support System Demo")
    print("🎯 Demonstrating AI-powered train schedule optimization")
    print("📋 Based on Törnquist & Persson (2007) research paper")
    
    # Initial status
    print_step(0, "Initial System Status")
    
    # Start simulation
    print("🏁 Starting simulation...")
    result = make_request("/start_sim", "POST", {})
    if not result:
        print("❌ Failed to start simulation. Make sure the server is running.")
        return False
    
    time.sleep(1)
    show_system_status()
    
    # Demo workflow
    if not demonstrate_disruption():
        return False
    
    time.sleep(2)
    
    optimization_result = run_optimization()
    if not optimization_result:
        return False
    
    time.sleep(1)
    
    pending_decisions = show_pending_decisions()
    
    # Approve a high-impact decision if available
    if pending_decisions:
        print_step(3.5, "Admin Decision Approval")
        
        # Find a high-impact decision
        high_impact_decision = None
        for decision in pending_decisions:
            impact = abs(decision.get('impact_analysis', {}).get('delay_change', 0))
            if impact > 3:  # More than 3 minutes impact
                high_impact_decision = decision
                break
        
        if high_impact_decision:
            print(f"👨‍💼 Approving high-impact decision: {high_impact_decision['decision_id']}")
            print(f"   Impact: {high_impact_decision.get('impact_analysis', {}).get('delay_change', 0):.1f} minutes")
            
            if approve_decision(high_impact_decision['decision_id'], high_impact_decision['train_id']):
                time.sleep(1)
    
    show_optimization_results()
    
    print_header("Demo Completed Successfully!")
    print("✅ MILP optimization system demonstrated key capabilities:")
    print("   🧮 Mathematical optimization using proven MILP model")
    print("   🤖 Intelligent decision generation and auto-approval")
    print("   👨‍💼 Admin approval workflow for critical decisions")
    print("   📊 Real-time performance monitoring and recommendations")
    print("   ⚡ Significant delay reduction through AI optimization")
    
    return True

if __name__ == "__main__":
    print("🚂 Starting MILP Railway DSS Demo")
    print("📋 Make sure the Flask server is running on port 5000")
    print()
    
    try:
        success = main()
        if success:
            print("\n🌟 Demo completed successfully!")
            print("💡 The system is ready for SIH presentation!")
        else:
            print("\n❌ Demo encountered issues. Check server status.")
        
    except KeyboardInterrupt:
        print("\n⏸️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        import traceback
        traceback.print_exc()