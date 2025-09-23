#!/usr/bin/env python3
"""Debug script to test decision approval"""

import requests
import json

def test_approval():
    print("🔍 Testing MILP Decision Approval System")
    print("=" * 50)
    
    # Get pending decisions
    print("1. Getting pending decisions...")
    response = requests.get('http://localhost:5000/pending_decisions')
    data = response.json()
    
    if not data.get('success'):
        print("❌ Failed to get pending decisions")
        return
        
    decisions = data.get('decisions', [])
    print(f"✅ Found {len(decisions)} pending decisions")
    
    if not decisions:
        print("❌ No decisions to test")
        return
        
    # Test the first decision
    decision = decisions[0]
    decision_id = decision['id']
    print(f"\n2. Testing approval for decision: {decision_id}")
    print(f"   Type: {decision['type']}")
    print(f"   Train: {decision['train_id']}")
    print(f"   Station: {decision['station']}")
    print(f"   Description: {decision['description']}")
    
    # Test approval
    print(f"\n3. Attempting to approve decision...")
    approval_response = requests.post('http://localhost:5000/approve_decision', 
                                     json={'decision_id': decision_id})
    
    print(f"   Status Code: {approval_response.status_code}")
    result = approval_response.json()
    print(f"   Success: {result.get('success')}")
    print(f"   Message: {result.get('message')}")
    
    if result.get('success'):
        print("✅ Approval successful!")
        print(f"   Train ID: {result.get('train_id')}")
        print(f"   Decision Type: {result.get('decision_type')}")
    else:
        print("❌ Approval failed!")
        
    print("\n4. Checking if decision was removed from pending...")
    response2 = requests.get('http://localhost:5000/pending_decisions')
    data2 = response2.json()
    remaining_decisions = data2.get('decisions', [])
    decision_ids = [d['id'] for d in remaining_decisions]
    
    if decision_id in decision_ids:
        print(f"❌ Decision {decision_id} still in pending list")
    else:
        print(f"✅ Decision {decision_id} removed from pending list")

if __name__ == "__main__":
    test_approval()