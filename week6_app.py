"""Small human protein-network explorer using STRING v12.0's API.

Run with: streamlit run week6_app.py
The app fetches only a small network neighborhood (not STRING's full database).
"""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components


st.set_page_config(page_title="Protein Network Explorer", page_icon="🧬", layout="wide")
#设置网页的标签（标题、图表、页面布局）
STRING_API = "https://version-12-0.string-db.org/api/tsv/network"
STRING_DOWNLOADS = "https://version-12-0.string-db.org/cgi/download?species_text=Homo+sapiens"
STRING_LICENSE = "https://version-12.string-db.org/cgi/access"
#数据来源

PRESETS = {
    "Cancer signaling (default)": [
        "TP53", "EGFR", "BRCA1", "MYC", "AKT1", "MDM2", "CDK2", "RB1",
        "ATM", "KRAS", "BRAF", "PIK3CA",
    ],
    "Immune response": [
        "IL6", "TNF", "IL1B", "IFNG", "STAT1", "STAT3", "NFKB1", "RELA",
        "TLR4", "MYD88", "JAK1", "JAK2", "CXCL8", "CXCR4", "CCL2", "CD4",
        "CD8A", "CD3D", "CD3E", "FOXP3", "CTLA4", "PDCD1", "CD274", "TGFB1",
    ],
    "Cell cycle": [
        "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1", "CCNE1",
        "RB1", "TP53", "CDKN1A", "CDKN1B", "E2F1", "MKI67", "ATM", "ATR",
        "CHEK1", "CHEK2", "CDC25A", "CDC25C", "PLK1", "AURKA", "AURKB", "MDM2",
    ],
}
#三个种类的蛋白质，能提供给用户进行选择


@st.cache_data(show_spinner="Fetching a small protein neighborhood from STRING…")
#缓存装饰器
def fetch_network(
    protein_names: tuple[str, ...], required_score: int, add_nodes: int
) -> pd.DataFrame:
    """Request only a selected protein set and a limited number of neighbors."""
    params = urllib.parse.urlencode(
        {
            "identifiers": "\r".join(protein_names),
            "species": 9606,
            "required_score": required_score,
            "add_nodes": add_nodes,
            "caller_identity": "protein_network_explorer_coursework",
        }
    )
    #构造API查询参数
    
    request = urllib.request.Request(
        f"{STRING_API}?{params}",
        headers={"User-Agent": "ProteinNetworkExplorer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        text = response.read().decode("utf-8")
    if not text.strip():
        return pd.DataFrame(
            columns=["stringId_A", "stringId_B", "preferredName_A", "preferredName_B", "score"]
        )
    return pd.read_csv(io.StringIO(text), sep="\t")
#获取数据


@st.cache_data(show_spinner=False)
def make_graph(edge_df: pd.DataFrame, min_score: int) -> tuple[nx.Graph, dict[str, str]]:
    """Filter STRING associations and create a labeled undirected graph."""
    G = nx.Graph()#创建空的无向图
    names: dict[str, str] = {}#创建空的名称字典
    for row in edge_df.itertuples(index=False):
        score = float(row.score)
        if score < min_score / 1000: #用于过滤
            continue
        u, v = str(row.stringId_A), str(row.stringId_B)
        name_u, name_v = str(row.preferredName_A), str(row.preferredName_B)
        #读取蛋白质ID和名称
        names[u], names[v] = name_u, name_v
        G.add_edge(u, v, score=score)#作为节点加入图中，并保留置信度
    nx.set_node_attributes(G, names, "label")
    return G, names


def detect_communities(G: nx.Graph) -> dict[str, int]:
    if G.number_of_nodes() == 0:
        return {}
    groups = nx.community.greedy_modularity_communities(G, weight=None)
    #用贪心模块度算法把节点分成若干社区
    return {node: idx for idx, group in enumerate(groups) for node in group}


def main() -> None:
    st.title("🧬 Human Protein Network Explorer")
    st.caption("A compact, interactive STRING protein-association network — built from a small query, not a full-database download.")
    st.caption("Jinyi Zhou")
    with st.sidebar:#侧边栏控件
        st.header("Explore the network")
        preset = st.selectbox("Protein set", list(PRESETS)) #选择数据集
        selected_names = PRESETS[preset]
        st.caption(f"Seed list: {len(selected_names)} proteins")
        add_nodes = st.slider(
            "Additional high-confidence neighbors",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="STRING adds up to this many interaction partners around the selected seed proteins.",
        )
        
        confidence = st.slider(
            "Minimum confidence score",
            min_value=600,
            max_value=900,
            value=800,
            step=25,
            help="STRING combined confidence is on a 0–1000 scale.",
        )
        
        max_display_nodes = st.slider(
            "Maximum proteins displayed",
            min_value=8,
            max_value=40,
            value=16,
            step=2,
            help="If the network is still crowded, only the most connected proteins are retained.",
        )
        
        layout = st.selectbox("Layout", ["Force-directed (spring)", "Circular"])
        centrality_name = st.selectbox("Centrality filter", ["Degree", "Betweenness", "Closeness"])
        percentile = st.slider("Keep nodes at/above centrality percentile", 0, 90, 25, 5)#百分位门栏
        show_matrix = st.checkbox("Show adjacency matrix", value=True)

    try:
        api_edges = fetch_network(tuple(selected_names), confidence, add_nodes)#调用API
        G, labels = make_graph(api_edges, confidence)# 转成图和名称字典
    except Exception as exc:
        st.error(f"STRING API request failed: {exc}")
        st.info("Check your internet connection and rerun the app. The app requests only this small protein network.")
        st.stop()

    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        st.warning("No associations remain at this score. Lower the confidence threshold or choose another protein set.")
        st.stop()

    degree = dict(G.degree()) #计算每个节点的度数
    communities = detect_communities(G)#检测社区
    if centrality_name == "Degree":
        centrality = nx.degree_centrality(G)
    elif centrality_name == "Betweenness":
        centrality = nx.betweenness_centrality(G, normalized=True)
    else:
        centrality = nx.closeness_centrality(G)
    #根据不同的选择计算不同的中心性
    
    cutoff = float(pd.Series(centrality).quantile(percentile / 100))#计算中心性数值的百分位门栏
    keep_nodes = {n for n, value in centrality.items() if value >= cutoff}#保留符合的节点（大于所选择的百分位）
    # Apply an explicit node cap after the centrality threshold. Degree is used
    # for the cap so the most connected proteins stay visible and the graph
    # cannot silently grow into a hairball.
    if len(keep_nodes) > max_display_nodes:#过滤不想让太多的点进行展示
        keep_nodes = set(sorted(keep_nodes, key=lambda n: (degree[n], centrality[n]), reverse=True)[:max_display_nodes])
    H = G.subgraph(keep_nodes).copy()
    H.remove_nodes_from(list(nx.isolates(H)))#删除孤立节点
    if H.number_of_nodes() < 2:
        st.warning("This centrality cutoff leaves too few connected proteins. Lower the percentile.")
        st.stop()
    shown_communities = detect_communities(H)

    st.markdown(
        """
        **What this network represents.** Each node is a human protein and each edge is a STRING functional association supported by combined evidence. Associations may be supported by experiments, curated databases, co-expression, or text mining; an edge does **not necessarily mean direct physical binding**.

        **Why these encodings.** Node size maps to degree, so larger nodes have more direct associations in the displayed network. Node color maps to detected communities, highlighting groups with dense internal connections. Edge width maps to STRING's combined confidence score.

        **Limitation.** This is a small, query-centered sample of the human protein network, not the complete interactome. STRING's score measures confidence in an association, not interaction strength or proof of direct binding. A force-directed layout is algorithmic, so visual proximity is not a measured biological distance.
        """
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Proteins shown", H.number_of_nodes())
    m2.metric("Associations shown", H.number_of_edges())
    m3.metric("Detected communities", len(set(shown_communities.values())))
    m4.metric("Confidence cutoff", confidence)

    st.subheader("Interactive node-link view")
    st.caption(
        f"Size = degree · Color = detected community · Edge width = confidence · "
        f"Filtered by {centrality_name.lower()} (≥ {percentile}th percentile)"
    )
    if layout == "Circular":
        positions = nx.circular_layout(H)
    else:
        positions = nx.spring_layout(H, seed=42)

    palette = ["#2878A5", "#2A9D8F", "#E9A23B", "#8E6BBE", "#D95F59", "#58A55C", "#D77FA1", "#6B7280"]
    max_degree = max(degree[n] for n in H.nodes) or 1
    net = Network(height="600px", width="100%", directed=False, bgcolor="#ffffff", font_color="#25364a")
    #创建PyVis网络对象
    physics_enabled = layout != "Circular"
    net.set_options(
        json.dumps(
            {
                "nodes": {
                    "font": {"size": 15, "face": "Arial"},
                    "scaling": {"min": 10, "max": 24},
                },
                "edges": {
                    "smooth": {"enabled": False},
                    "color": {"color": "#9aa8b3", "highlight": "#2878A5"},
                },
                "physics": {
                    "enabled": physics_enabled,
                    "barnesHut": {
                        "gravitationalConstant": -5000,
                        "centralGravity": 0.25,
                        "springLength": 145,
                        "springConstant": 0.025,
                    },
                    "minVelocity": 0.75,
                },
            }
        )
    )
    #显示参数
    for node in H.nodes:
        d = degree[node]
        community = shown_communities.get(node, 0)
        xy = positions[node]
        net.add_node(
            node,
            label=labels[node],
            title=f"{labels[node]}<br>Degree: {d}<br>{centrality_name}: {centrality[node]:.3f}<br>Community: {community + 1}",
            size=10 + 14 * d / max_degree,
            color=palette[community % len(palette)],
            x=float(xy[0] * 700),
            y=float(xy[1] * 500),
            physics=(layout != "Circular"),
        )
    #逐个添加网络节点
    for u, v, data in H.edges(data=True):
        score = data["score"]
        net.add_edge(u, v, value=max(1, score * 3), title=f"STRING confidence: {score:.3f}")
    components.html(net.generate_html(notebook=False), height=620, scrolling=True)
    #添加边

    if show_matrix:
        st.subheader("Adjacency matrix")
        st.caption("The same filtered proteins and associations in matrix form. Rows and columns are ordered by detected community.")
        matrix_nodes = sorted(H.nodes, key=lambda n: (shown_communities[n], labels[n]))
        A = nx.to_numpy_array(H, nodelist=matrix_nodes)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            A,
            cmap=sns.color_palette(["#F3F7FA", "#2878A5"], as_cmap=True),
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
            xticklabels=[labels[n] for n in matrix_nodes],
            yticklabels=[labels[n] for n in matrix_nodes],
            ax=ax,
        )
        ax.set_xlabel("Protein")
        ax.set_ylabel("Protein")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0, labelsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            "**Readability comparison:** the node-link view makes hubs and neighborhood shape easier to interpret. The matrix avoids overlapping edges and makes dense blocks easier to compare, but following an individual protein's path is less intuitive."
        )

    with st.expander("Dataset source and attribution"):
        st.markdown(
            f"""
            - **Source:** [STRING v12.0 API and downloads]({STRING_DOWNLOADS})
            - **Organism:** *Homo sapiens* (NCBI taxonomy ID 9606)
            - **Query:** {len(selected_names)} seed proteins from the **{preset}** set, plus up to {add_nodes} high-confidence neighbors. The visualization is capped at {max_display_nodes} proteins.
            - **Returned fields:** protein IDs, preferred names, and combined association score.
            - **License:** [CC BY 4.0]({STRING_LICENSE}). Please cite STRING and preserve attribution when reusing the data.
            """
        )


if __name__ == "__main__":
    main()
