import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
from scipy.stats import linregress
matplotlib.use('TkAgg')

# טוען את נתוני CORA
def load_cora():
    content_df = pd.read_csv("cora.content", sep='\t', header=None)
    content_df.columns = ['id'] + [f'word_{i}' for i in range(1, 1434)] + ['category']
    cites_df = pd.read_csv("cora.cites", sep='\t', header=None, names=['target', 'source'])
    return content_df, cites_df

# טוען את נתוני משחקי הכס
def load_thrones():
    df = pd.read_csv("stormofswords.csv")
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
    return G

# בונה גרף CORA עם משקלים רנדומליים
def build_cora_graph_with_weights(content_df, cites_df):
    G = nx.DiGraph()
    for _, row in content_df.iterrows():
        G.add_node(row['id'], category=row['category'])
    for _, row in cites_df.iterrows():
        if row['source'] in G and row['target'] in G:
            weight = random.randint(1, 10)  # משקל רנדומלי
            G.add_edge(row['source'], row['target'], weight=weight)
    return G

# חישוב חפיפת שכונות
def calculate_neighborhood_overlap(G):
    overlaps = []
    weights = []

    for u, v, data in G.edges(data=True):
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))

        # חפיפת השכונות
        common = neighbors_u & neighbors_v
        union = neighbors_u | neighbors_v

        overlap = len(common) / len(union) if len(union) > 0 else 0

        overlaps.append(overlap)
        weights.append(data['weight'])

    return overlaps, weights

# יצירת גרף חפיפת שכונות כמשקל פונקציה
def plot_neighborhood_overlap(G, title, filename):
    overlaps, weights = calculate_neighborhood_overlap(G)

    # חישוב קו מגמה לינארי
    slope, intercept, _, _, _ = linregress(weights, overlaps)
    trend_y = [slope * w + intercept for w in weights]

    # יצירת גרף
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

# הסרת קשתות ומעקב אחרי גודל רכיב הענק
def edge_removal_simulation(G, color, title, filename):
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    sizes = []  # גודל הרכיב הקשיר הגדול ביותר אחרי כל הסרה
    removed = []  # מספר הקשתות שהוסרו
    G_copy = G.copy()  # יצירת עותק של הגרף כדי לשמור על הגרף המקורי
    step = max(1, len(edges_sorted) // 100)  # כל פעם נסיר מספר קשתות אחיד

    # חישוב גודל רכיב הענק אחרי כל הסרה
    for i in range(0, len(edges_sorted), step):
        if nx.is_empty(G_copy):
            sizes.append(0)
        else:
            sizes.append(len(max(nx.connected_components(G_copy), key=len)))
        removed.append(i)

        # הסר את הקשתות מהגרף
        for edge in edges_sorted[i:i+step]:
            G_copy.remove_edge(edge[0], edge[1])

    # יצירת גרף שמציג את השפעת הסרת הקשתות
    plt.figure(figsize=(10, 6))
    plt.plot(removed, sizes, marker='o', linestyle='-', linewidth=2, markersize=6,
             color=color, markerfacecolor='white', markeredgewidth=1.5)
    plt.xlabel("Number of Edges Removed", fontsize=12)
    plt.ylabel("Size of Largest Connected Component", fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# טעינת נתוני CORA
content_df, cites_df = load_cora()
G_cora = build_cora_graph_with_weights(content_df, cites_df)
G_cora_undirected = G_cora.to_undirected()

# טעינת נתוני משחקי הכס
G_thrones = load_thrones()

# עבור CORA - חפיפת שכונות
plot_neighborhood_overlap(G_cora, "CORA – Neighborhood Overlap vs Weight", "cora_neighborhood_overlap.png")

# עבור Network of Thrones - חפיפת שכונות
plot_neighborhood_overlap(G_thrones, "Thrones – Neighborhood Overlap vs Weight", "thrones_neighborhood_overlap.png")

# עבור CORA - הסרת קשתות
edge_removal_simulation(G_cora_undirected, "#1f77b4", "CORA – Edge Removal Impact", "cora_edge_removal.png")

# עבור Network of Thrones - הסרת קשתות
edge_removal_simulation(G_thrones, "#7b2cbf", "Thrones – Edge Removal Impact", "thrones_edge_removal.png")
