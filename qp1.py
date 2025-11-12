import pandas as pd
import random
import heapq
import networkx as nx
import matplotlib.pyplot as plt

sections = [f"S{i}" for i in range(1, 15)]  
data = []
for i in range(len(sections) - 1):
    for j in range(i + 1, len(sections)):
        if random.random() < 0.3: 
            data.append(
                {
                    "From": sections[i],
                    "To": sections[j],
                    "Distance": random.randint(5, 25),
                }
            )

df = pd.DataFrame(data)
df.to_csv("warehouse_routes.csv", index=False)

heuristics = {s: random.randint(1, 20) for s in sections}

def build_graph(filename):
    data = pd.read_csv(filename)
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row["From"], row["To"], weight=row["Distance"])
    return G

G = build_graph("warehouse_routes.csv")

def astar_with_count_simple(G, start, goal, heuristics):
    pq = [(heuristics[start], start, [start], 0)]
    visited = set()
    explored = 0

    while pq:
        est, node, path, cost = heapq.heappop(pq)
        explored += 1

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, cost, explored

        for neighbor in G.neighbors(node):
            dist = G[node][neighbor]["weight"]
            new_cost = cost + dist
            est_total = new_cost + heuristics[neighbor]
            heapq.heappush(pq, (est_total, neighbor, path + [neighbor], new_cost))

    return None, float("inf"), explored

for node in G.nodes():
    neighbors = [(n, G[node][n]["weight"]) for n in G.neighbors(node)]
    print(f"{node} -> {neighbors}")

matrix = nx.to_pandas_adjacency(G, weight="weight")

def visualize_graph_simple(G, heuristics, path=None):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=[heuristics[n] for n in G.nodes()],
        cmap=plt.cm.Blues,
        node_size=800,
        font_weight="bold",
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3)

    plt.title("Warehouse Graph (Node Color = Heuristic Value)")
    plt.axis("off")
    plt.show()

start_node = "S1"
goal_node = "S10"
path, cost, explored = astar_with_count_simple(G, start_node, goal_node, heuristics)
visualize_graph_simple(G, heuristics, path)