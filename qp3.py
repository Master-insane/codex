import pandas as pd
import random
import heapq
import networkx as nx
import matplotlib.pyplot as plt

intersections = [f"I{i}" for i in range(1, 18)]
data = []

for i in range(len(intersections)):
    for j in range(len(intersections)):
        if i != j and random.random() < 0.2:  # 20% chance of a road
            travel_time = random.randint(2, 15)
            data.append(
                {
                    "From": intersections[i],
                    "To": intersections[j],
                    "Travel_Time": travel_time,
                }
            )

df = pd.DataFrame(data)
df.to_csv("smartcity_roads.csv", index=False)

heuristics = {node: random.randint(1, 20) for node in intersections}


def build_graph(filename):
    data = pd.read_csv(filename)
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row["From"], row["To"], weight=row["Travel_Time"])
    return G


G = build_graph("smartcity_roads.csv")
for node in G.nodes():
    neighbors = [(n, G[node][n]["weight"]) for n in G.neighbors(node)]
    print(f"{node} -> {neighbors}")

print(nx.to_pandas_adjacency(G, weight="weight"))


def astar_search(G, start, goal, heuristics):
    pq = [(heuristics[start], start, [start], 0)]  # (f, node, path, g)
    visited = set()
    explored = 0

    while pq:
        pq.sort(key=lambda x: x[0])
        f, node, path, cost = pq.pop(0)
        explored += 1

        if node == goal:
            return path, cost, explored

        if node in visited:
            continue
        visited.add(node)

        for neighbor in G.neighbors(node):
            travel = G[node][neighbor]["weight"]
            new_cost = cost + travel
            f_score = new_cost + heuristics.get(neighbor, 0)
            pq.append((f_score, neighbor, path + [neighbor], new_cost))

    return None, float("inf"), explored


start, goal = "I1", "I10"
path, cost, explored = astar_search(G, start, goal, heuristics)


def near_optimal_routes(G, start, goal, heuristics, factor=1.2):
    best_path, best_cost, _ = astar_search(G, start, goal, heuristics)
    if not best_path:
        return []

    limit = best_cost * factor
    routes = []

    def explore(node, path, cost):
        if cost > limit:
            return
        if node == goal:
            routes.append((path, cost))
            return
        for neighbor in G.neighbors(node):
            if neighbor not in path:
                travel = G[node][neighbor]["weight"]
                explore(neighbor, path + [neighbor], cost + travel)

    explore(start, [start], 0)
    return [(p, c) for p, c in routes if c <= limit]


near_routes = near_optimal_routes(G, start, goal, heuristics)

for p, c in near_routes:
    print(f"  Path: {p}, Travel time: {c}")


def check_consistency(G, heuristics):
    bad_edges = []
    for u, v, d in G.edges(data=True):
        cost = d["weight"]
        if heuristics[u] > cost + heuristics[v]:
            bad_edges.append((u, v, heuristics[u], cost, heuristics[v]))
    return bad_edges


violations = check_consistency(G, heuristics)
if not violations:
    print("✅ Heuristic is consistent for all edges.")
else:
    print("⚠️ Inconsistent heuristic edges:")
    for u, v, hu, c, hv in violations:
        print(f"  {u}->{v}: h({u})={hu}, cost={c}, h({v})={hv}")


def visualize_graph(G, heuristics, path=None):
    pos = nx.spring_layout(G, seed=42)
    node_colors = [heuristics[n] for n in G.nodes()]

    # Draw nodes (color = heuristic value)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.Blues,
        node_size=800,
        font_weight="bold",
    )

    # Label edges with travel times
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Highlight the optimal path
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="r", width=3)

    plt.title("Smart City Road Network (A* Optimal Path Highlighted)")
    plt.axis("off")
    plt.show()


# Show graph
visualize_graph(G, heuristics, path)