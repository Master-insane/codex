import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

buildings = [
    'Admin Block', 'Library', 'Hostel A', 'Hostel B', 'Cafeteria', 'Auditorium',
    'Sports Complex', 'Lab 1', 'Lab 2', 'Workshop', 'Bookstore', 'Parking Lot'
]

data = []
for i in range(1, len(buildings)):
    parent = buildings[random.randint(0, i - 1)]
    child = buildings[i]
    data.append({'From_Building': parent, 'To_Building': child})

df = pd.DataFrame(data)
df.to_csv('campus_routes.csv', index=False)

print("✅ Dataset 'campus_routes.csv' created!")
print(df.head())
print("----------------------------------------------------------")

def build_graph(filename):
    data = pd.read_csv(filename)
    G = nx.DiGraph()

    for _, row in data.iterrows():
        G.add_edge(row['From_Building'], row['To_Building'])
    return G

G = build_graph('campus_routes.csv')

print("🏗️ Campus Buildings and Connections:")
for node in G.nodes():
    print(f"{node}: {list(G.successors(node))}")
print("----------------------------------------------------------")

def bfs_shortest_path(G, start, goal):
    visited = set([start])
    queue = deque([[start]])

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path  # shortest path found

        for neighbor in G.successors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None

start, goal = 'Admin Block', 'Parking Lot'
path = bfs_shortest_path(G, start, goal)
if path:
    print(f"🚚 Shortest delivery route from {start} to {goal}: {path}")
else:
    print(f"❌ No route from {start} to {goal}.")

for node in G.nodes():
    G.nodes[node]['load'] = random.randint(1, 10)

print("📦 Delivery load factors (per building):")
for node, data in G.nodes(data=True):
    print(f"{node}: {data['load']}")

def bfs_min_load_path(G, start, goal):
    queue = deque([(start, [start], G.nodes[start]['load'])])
    best_path, best_load = None, float('inf')

    while queue:
        node, path, load = queue.popleft()

        if node == goal:
            if load < best_load:
                best_path, best_load = path, load
            continue

        for neighbor in G.successors(node):
            if neighbor not in path:
                total_load = load + G.nodes[neighbor]['load']
                queue.append((neighbor, path + [neighbor], total_load))

    return best_path, best_load

min_path, min_load = bfs_min_load_path(G, start, goal)
print(f"🪣 Minimum-load path from {start} to {goal}: {min_path}")
print(f"Total delivery load: {min_load}")
print("----------------------------------------------------------")

def bfs_reachable(G, start):
    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in G.successors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

reachable = bfs_reachable(G, 'Admin Block')
print(f"🏢 Buildings reachable from Admin Block: {reachable}")
print(f"Total reachable buildings: {len(reachable)}")
print("----------------------------------------------------------")

def visualize_graph(G, path=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    colors = ['red' if path and n in path else 'skyblue' for n in G.nodes()]

    nx.draw(
        G, pos, with_labels=True, node_color=colors,
        node_size=1000, font_size=9, arrows=True, font_weight='bold'
    )
    if path:
        edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=3)

    plt.title("Campus Logistics Network (Directed)")
    plt.axis('off')
    plt.show()

visualize_graph(G, path)
def has_cycle(G):
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in G.successors(node):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in G.nodes():
        if node not in visited:
            if dfs(node):
                return True
    return False

if has_cycle(G):
    print("⚠️ Cycle detected! There may be a routing loop.")
else:
    print("✅ No cycles detected in the logistics network.")