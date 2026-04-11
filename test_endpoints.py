import requests

s = requests.Session()
base = "http://localhost:7860"

# Test health
print("--- /health ---")
r = s.get(f"{base}/health")
print(f"  Status: {r.status_code}")
print(f"  Body: {r.json()}")

# Test reset
print("\n--- /reset ---")
r = s.post(f"{base}/reset", json={"task_id": "easy_unlimited_liability"})
print(f"  Status: {r.status_code}")
data = r.json()
print(f"  Keys: {list(data.keys())}")
obs = data.get("observation", {})
print(f"  clause_type: {obs.get('clause_type')}")
print(f"  risk_level: {obs.get('risk_level')}")
print(f"  done: {obs.get('done')}")

# Test step
print("\n--- /step ---")
action = {"action_type": "FLAG_RISK"}
r = s.post(f"{base}/step", json={"action": action})
print(f"  Status: {r.status_code}")
data = r.json()
print(f"  Keys: {list(data.keys())}")
print(f"  reward: {data.get('reward')}")
print(f"  done: {data.get('done')}")

# Test schema
print("\n--- /schema ---")
r = s.get(f"{base}/schema")
print(f"  Status: {r.status_code}")
print(f"  Keys: {list(r.json().keys())}")

# Test state
print("\n--- /state ---")
r = s.get(f"{base}/state")
print(f"  Status: {r.status_code}")
print(f"  Body: {r.json()}")

print("\nAll HTTP endpoints OK!")
