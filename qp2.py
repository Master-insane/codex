import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

teams = ['Analytics', 'Design', 'Support', 'Development', 'QA', 'Management']
channels = [f'Channel_{i}' for i in range(1, 16)]

data = []
for channel in channels:
    communicating_teams = random.sample(teams, random.randint(2, 4))
    for t in communicating_teams:
        data.append({'Team': t, 'Channel': channel})

df = pd.DataFrame(data)
df.to_csv('team_comm.csv', index=False)

print("✅ Dataset 'team_comm.csv' created!")
print(df.head())
print("----------------------------------------------------------")

def build_graph(filename):
    data = pd.read_csv(filename)
    G = nx.Graph()
    for channel, group in data.groupby('Channel'):
        members = group['Team'].tolist()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                G.add_edge(members[i], members[j], channel=channel)
    return G

G = build_graph('team_comm.csv')
print("📡 Team Communication Links:")
for team in G.nodes():
    print(f"{team}: {G.degree(team)} links")
def bfs_shortest_path(G, start, goal):
    visited = set([start])
    queue = deque([[start]])

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None
start, goal = 'Analytics', 'QA'
path = bfs_shortest_path(G, start, goal)
if path:
    print(f"🔗 Shortest communication chain from {start} to {goal}: {path}")
else:
    print(f"❌ No communication chain between {start} and {goal}.")

def bfs_reachable(G, source):
    visited = set([source])
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited

reachable = bfs_reachable(G, 'Analytics')
print(f"🌐 Teams reachable from Analytics: {reachable}")
print(f"Total reachable teams: {len(reachable)}")
print("----------------------------------------------------------")

# ----------------------------------------------------------
# (d) Adjacency Matrix
# ----------------------------------------------------------
adj_matrix = nx.to_pandas_adjacency(G, dtype=int)
print("🧮 Adjacency Matrix:")
print(adj_matrix)
print("----------------------------------------------------------")

# ----------------------------------------------------------
# (e) Visualization
# ----------------------------------------------------------
def visualize_graph(G, path=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    # Node colors: highlight those in the shortest path
    color_map = ['red' if path and node in path else 'skyblue' for node in G.nodes()]

    nx.draw(
        G, pos, with_labels=True, node_color=color_map,
        node_size=1000, font_weight='bold', font_size=10
    )

    # Highlight path edges
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=3)

    plt.title("Organizational Communication Network (Shortest Chain Highlighted)")
    plt.axis('off')
    plt.show()

# Show graph
visualize_graph(G, path)