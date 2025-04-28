import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
from scipy.stats import linregress
import copy

matplotlib.use('TkAgg')
def load_thrones():
    df = pd.read_csv("stormofswords.csv")
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
    return G
# חישוב Neighborhood Overlap לכל קשת
def calculate_overlap_for_edges(G):
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

    return np.array(weights), np.array(overlaps)


# גרף אחוזון חפיפה
def plot_percentile_overlap(G, title, filename):
    weights, overlaps = calculate_overlap_for_edges(G)

    # מיון לפי משקל
    sorted_indices = np.argsort(weights)
    sorted_weights = weights[sorted_indices]
    sorted_overlaps = overlaps[sorted_indices]

    percentiles = np.linspace(0, 1, len(sorted_weights))

    plt.figure(figsize=(8, 6))
    plt.plot(percentiles, sorted_overlaps, 'o', markersize=3, label='Edges')

    # קו מגמה חלק (moving average קטן)
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(sorted_overlaps, size=10)
    plt.plot(percentiles, smoothed, '-', color='red', linewidth=2, label='Trend')

    plt.xlabel(r'$P_{cum}(w)$ (Strength Percentile)', fontsize=12)
    plt.ylabel(r'$\langle O \rangle_w$ (Neighborhood Overlap)', fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

G_thrones = load_thrones()

# להריץ על רשת Thrones:
plot_percentile_overlap(G_thrones, "Thrones – Percentile of Edge Weight vs Neighborhood Overlap",
                        "thrones_percentile_overlap.png")
