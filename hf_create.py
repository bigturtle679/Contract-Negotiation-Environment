"""
Helper script to create the HuggingFace Space.
Run once to set up. Requires HF_TOKEN environment variable.
"""
import urllib.request
import json
import os

token = os.getenv("HF_TOKEN")
if not token:
    print("ERROR: Set HF_TOKEN environment variable first.")
    raise SystemExit(1)

url = "https://huggingface.co/api/repos/create"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
data = json.dumps({
    "name": "Contract-Negotiation-Environment",
    "type": "space",
    "sdk": "docker",
    "private": False,
}).encode("utf-8")

req = urllib.request.Request(url, headers=headers, data=data)
try:
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode("utf-8"))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Failed:", e)
