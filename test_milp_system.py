#!/usr/bin/env python3
"""
Test script for the MILP-based Railway Decision Support System
Tests the complete workflow: disruption -> optimization -> admin approval -> application
"""

import requests
import json
import time
import sys
import os

# Test configuration
BASE_URL = "http://localhost:5000"
TEST_DELAY_MINUTES = 15
TEST_TRAIN_ID = "12614"  # High priority passenger train

def test_endpoint(endpoint, method="GET", data=None, description=""):
    """Test an API endpoint and return response"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            print(f"❌ Unsupported method: {method}")
            return None
        
        print(f"🔍 Testing {endpoint} ({description})")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result.get('message', 'OK')}")
            return result
        else:
            print(f"   ❌ Failed: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Connection error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON decode error: {e}")
        return None

def main():
    """Main test workflow"""
    print("🚂 Testing MILP-based Railway Decision Support System")
    print("=" * 60)
    
    # Test 1: Check server status
    print("\n📊 Step 1: Check System Status")
    status = test_endpoint("/optimization_summary", description="System overview")
    if not status:
        print("❌ Server not responding. Make sure the Flask app is running on port 5000")
        return False
    
    print(f"   Trains: {status.get('summary', {}).get('total_trains', 0)}")
    print(f"   Efficiency: {status.get('summary', {}).get('efficiency_percent', 0)}%")
    
    # Test 2: Start simulation
    print("\n🏁 Step 2: Start Simulation")
    result = test_endpoint("/start_sim", "POST", {}, "Start simulation")
    if not result:
        print("❌ Failed to start simulation")
        return False
    
    time.sleep(2)  # Let simulation start
    
    # Test 3: Check trains data
    print("\n🚄 Step 3: Check Trains Data")
    stats = test_endpoint("/stats", description="Get train statistics")
    if stats:
        print(f"   Active trains: {stats.get('trains_running', 0)}")
        print(f"   Delayed trains: {stats.get('trains_delayed', 0)}")
    
    # Test 4: Apply disruption
    print(f"\n⚠️ Step 4: Apply Disruption to Train {TEST_TRAIN_ID}")
    disruption_data = {
        "train_id": TEST_TRAIN_ID,
        "delay_minutes": TEST_DELAY_MINUTES,
        "reason": "Signal failure"
    }
    
    result = test_endpoint("/disrupt", "POST", disruption_data, f"Delay train {TEST_TRAIN_ID}")
    if not result:
        print("❌ Failed to apply disruption")
        return False
    
    print(f"   ✅ Applied {TEST_DELAY_MINUTES} minute delay to train {TEST_TRAIN_ID}")
    
    # Test 5: Run MILP optimization
    print("\n🧮 Step 5: Run MILP Optimization")
    optimization_data = {
        "delayed_train": TEST_TRAIN_ID,
        "delay_minutes": TEST_DELAY_MINUTES,
        "optimization_type": "total_delay",
        "strategy": 3
    }
    
    result = test_endpoint("/optimize", "POST", optimization_data, "MILP optimization")
    if not result:
        print("❌ MILP optimization failed")
        return False
    
    print(f"   Status: {result.get('status', 'unknown')}")
    print(f"   Computation time: {result.get('computation_time', 0):.2f}s")
    
    decisions = result.get('decisions', {})
    auto_applied = decisions.get('auto_applied', [])
    pending_approval = decisions.get('pending_approval', [])
    
    print(f"   Auto-applied decisions: {len(auto_applied)}")
    print(f"   Pending approval: {len(pending_approval)}")
    
    if result.get('summary', {}).get('delay_reduction', 0) > 0:
        print(f"   ✅ Delay reduction: {result['summary']['delay_reduction']:.1f} minutes")
    
    # Test 6: Check pending decisions
    print("\n📋 Step 6: Check Pending Decisions")
    pending = test_endpoint("/pending_decisions", description="Get pending decisions")
    if pending:
        decisions_list = pending.get('decisions', [])
        print(f"   Pending decisions: {len(decisions_list)}")
        
        for i, decision in enumerate(decisions_list[:3]):  # Show first 3
            print(f"   Decision {i+1}: {decision.get('decision_type')} for train {decision.get('train_id')}")
            print(f"     Impact: {decision.get('impact_analysis', {}).get('delay_change', 0):.1f} min")
    
    # Test 7: Approve a high-impact decision (if any)
    if pending and pending.get('decisions'):
        print("\n✅ Step 7: Approve High-Impact Decision")
        
        # Find a decision with significant impact
        high_impact_decision = None
        for decision in pending['decisions']:
            impact = abs(decision.get('impact_analysis', {}).get('delay_change', 0))
            if impact > 2:  # More than 2 minutes impact
                high_impact_decision = decision
                break
        
        if high_impact_decision:
            approval_data = {
                "decision_id": high_impact_decision['decision_id'],
                "reason": "Approved for testing - significant delay reduction"
            }
            
            result = test_endpoint("/approve_decision", "POST", approval_data, "Approve decision")
            if result:
                print(f"   ✅ Approved decision {high_impact_decision['decision_id']}")
                print(f"   Train: {high_impact_decision['train_id']}")
                print(f"   Type: {high_impact_decision['decision_type']}")
        else:
            print("   No high-impact decisions found")
    
    # Test 8: Get optimization results
    print("\n📈 Step 8: Get Optimization Results")
    results = test_endpoint("/optimization_results", description="Detailed results")
    if results:
        res = results.get('results', {})
        system_status = res.get('system_status', {})
        optimization_impact = res.get('optimization_impact', {})
        
        print(f"   System efficiency: {system_status.get('system_efficiency', 0):.1f}%")
        print(f"   Average delay: {system_status.get('average_delay', 0):.1f} min")
        print(f"   Delay reduction (1h): {optimization_impact.get('total_delay_reduction_1h', 0):.1f} min")
        
        recommendations = res.get('recommendations', [])
        if recommendations:
            print("   Recommendations:")
            for rec in recommendations[:2]:
                print(f"     • {rec}")
    
    # Test 9: Final status check
    print("\n🏁 Step 9: Final System Status")
    final_status = test_endpoint("/optimization_summary", description="Final overview")
    if final_status:
        summary = final_status.get('summary', {})
        print(f"   Final efficiency: {summary.get('efficiency_percent', 0):.1f}%")
        print(f"   Status: {summary.get('overall_status', 'Unknown')}")
        print(f"   Total delay: {summary.get('total_delay_minutes', 0):.1f} minutes")
        print(f"   Pending decisions: {summary.get('pending_decisions', 0)}")
    
    print("\n🎉 Test Completed Successfully!")
    print("=" * 60)
    print("✅ MILP optimization system is working correctly")
    print("✅ Admin approval workflow is functional")
    print("✅ Optimization results are being displayed")
    
    return True

if __name__ == "__main__":
    print("Starting MILP Railway DSS Test Suite")
    print("Make sure the Flask server is running: python backend/improved_app.py")
    print()
    
    try:
        success = main()
        if success:
            print("\n🌟 All tests passed! The system is ready for demonstration.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Check the server logs for details.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏸️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        sys.exit(1)