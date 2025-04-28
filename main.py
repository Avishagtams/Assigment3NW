import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
from scipy.stats import linregress
import copy

matplotlib.use('TkAgg')

# --- טוען את נתוני CORA
def load_cora():
    content_df = pd.read_csv("cora.content", sep='\t', header=None)
    content_df.columns = ['id'] + [f'word_{i}' for i in range(1, 1434)] + ['category']
    cites_df = pd.read_csv("cora.cites", sep='\t', header=None, names=['target', 'source'])
    return content_df, cites_df

# --- טוען את נתוני משחקי הכס
def load_thrones():
    df = pd.read_csv("stormofswords.csv")
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
    return G

# --- בונה גרף CORA עם משקלים רנדומליים
def build_cora_graph_with_weights(content_df, cites_df):
    G = nx.DiGraph()
    for _, row in content_df.iterrows():
        G.add_node(row['id'], category=row['category'])
    for _, row in cites_df.iterrows():
        if row['source'] in G and row['target'] in G:
            weight = random.randint(1, 10)  # משקל רנדומלי
            G.add_edge(row['source'], row['target'], weight=weight)
    return G

# --- חישוב חפיפת שכונות
def calculate_neighborhood_overlap(G):
    overlaps = []
    weights = []

    for u, v, data in G.edges(data=True):
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common = neighbors_u & neighbors_v
        union = neighbors_u | neighbors_v

        overlap = len(common) / len(union) if len(union) > 0 else 0

        overlaps.append(overlap)
        weights.append(data['weight'])

    return overlaps, weights

# --- יצירת גרף חפיפת שכונות
def plot_neighborhood_overlap(G, title, filename):
    overlaps, weights = calculate_neighborhood_overlap(G)

    slope, intercept, _, _, _ = linregress(weights, overlaps)
    trend_y = [slope * w + intercept for w in weights]

    plt.figure(figsize=(10, 6))
    plt.scatter(weights, overlaps, alpha=0.5, color='blue', edgecolors='k', linewidths=0.3, label='Edges')
    plt.plot(weights, trend_y, linestyle='--', color='green', linewidth=2, label='Trend Line')
    plt.xlabel("Edge Weight", fontsize=12)
    plt.ylabel("Neighborhood Overlap", fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# --- הסרת קשתות לפי כמה שיטות והשוואתן
def edge_removal_comparison(G, title, filename):
    def remove_edges(G, strategy):
        G_copy = copy.deepcopy(G)
        sizes = []
        removed = []

        if strategy == 'random':
            edges = list(G_copy.edges())
            random.shuffle(edges)
        elif strategy == 'weak':
            edges = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'])
        elif strategy == 'strong':
            edges = sorted(G_copy.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        elif strategy == 'betweenness':
            edge_betweenness = nx.edge_betweenness_centrality(G_copy)
            edges = sorted(edge_betweenness.keys(), key=lambda x: edge_betweenness[x], reverse=True)
        else:
            raise ValueError("Unknown strategy")

        step = max(1, len(edges) // 100)

        for i in range(0, len(edges), step):
            if nx.is_empty(G_copy):
                sizes.append(0)
            else:
                sizes.append(len(max(nx.connected_components(G_copy), key=len)))
            removed.append(i)

            for edge in edges[i:i+step]:
                if strategy == 'betweenness':
                    u, v = edge
                else:
                    u, v = edge[0], edge[1]
                if G_copy.has_edge(u, v):
                    G_copy.remove_edge(u, v)

        return removed, sizes

    strategies = {
        'Random': 'orange',
        'Weak edges first': 'blue',
        'Strong edges first': 'red',
        'Highest Betweenness': 'green'
    }

    plt.figure(figsize=(10, 6))

    for strategy_name, color in strategies.items():
        if strategy_name == 'Random':
            strategy_key = 'random'
        elif strategy_name == 'Weak edges first':
            strategy_key = 'weak'
        elif strategy_name == 'Strong edges first':
            strategy_key = 'strong'
        elif strategy_name == 'Highest Betweenness':
            strategy_key = 'betweenness'

        removed, sizes = remove_edges(G, strategy_key)
        plt.plot(removed, sizes, label=strategy_name, color=color, marker='o', markersize=4, linewidth=2)

    plt.xlabel("Number of Edges Removed", fontsize=12)
    plt.ylabel("Size of Largest Connected Component", fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# --- טעינת נתונים
content_df, cites_df = load_cora()
G_cora = build_cora_graph_with_weights(content_df, cites_df)
G_cora_undirected = G_cora.to_undirected()

G_thrones = load_thrones()

# --- ציור גרפי חפיפת שכונות
plot_neighborhood_overlap(G_cora, "CORA – Neighborhood Overlap vs Weight", "cora_neighborhood_overlap.png")
plot_neighborhood_overlap(G_thrones, "Thrones – Neighborhood Overlap vs Weight", "thrones_neighborhood_overlap.png")

# --- השוואת שיטות הסרת קשתות
edge_removal_comparison(G_cora_undirected, "CORA – Edge Removal Strategies Comparison", "cora_edge_removal_comparison.png")
edge_removal_comparison(G_thrones, "Thrones – Edge Removal Strategies Comparison", "thrones_edge_removal_comparison.png")
