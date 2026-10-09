import urllib.request, json

def get(url):
    req = urllib.request.Request('http://127.0.0.1:5000' + url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def post(url, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        'http://127.0.0.1:5000' + url,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

# Get current environment state
r = get('/environment')
print("=== CURRENT ENVIRONMENT STATE ===")
env = r.get('environment', {})
print("Grid:", env.get('grid_size'))
print("Depot:", env.get('depot'))
print("Bins:", list(env.get('bins', {}).keys()))
print("Bin positions:", env.get('bins'))
print("Obstacles count:", len(env.get('obstacles', [])))
print()
print("Q path length:", len(r.get('q_path', [])))
print("SARSA path length:", len(r.get('sarsa_path', [])))
print()
print("Q metrics:", r.get('q_metrics'))
print("SARSA metrics:", r.get('sarsa_metrics'))
print()

# Print the actual paths
qp = r.get('q_path', [])
sp = r.get('sarsa_path', [])
print("Q path (first 10):", qp[:10])
print("SARSA path (first 10):", sp[:10])
print()

# Take a step to see what happens
print("=== TAKING A STEP ===")
step_r = post('/step')
print("Step success:", step_r.get('success'))
print("Step message:", step_r.get('message'))
if step_r.get('success'):
    print("Position:", step_r.get('position'))
    print("Reward:", step_r.get('reward'))
    print("Steps:", step_r.get('steps'))
    print("Done:", step_r.get('done'))
    print("Collected bins:", step_r.get('collected_bins'))
    print("Q path length:", len(step_r.get('q_path', [])))
    print("SARSA path length:", len(step_r.get('sarsa_path', [])))
    print("Q metrics:", step_r.get('q_metrics'))
    print("SARSA metrics:", step_r.get('sarsa_metrics'))
