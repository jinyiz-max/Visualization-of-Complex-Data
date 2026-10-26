# -*- coding: utf-8 -*-

"""
Created on Tue Oct 20 17:35:29 2026

@author: ZJY
NYC 311 Signals — Multimodal Exploration Streamlit App
==========================================
Data source: NYC Open Data · 311 Service Requests from 2020 to Present
        Socrata ID: erm2-nwe9
        API: https://data.cityofnewyork.us/resource/erm2-nwe9.json

Tabs:
  Tab 1 · Map Explorer        → Map (Pydeck heatmap)
  Tab 2 · Time Pulse          → Time series + calendar heatmap (Plotly)
  Tab 3 · Language & Network  → Descriptor frequencies + complaint-descriptor co-occurrence network
  Tab 4 · Data Story          → KPI cards + narrative

"""

from datetime import date, timedelta
from io import BytesIO
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st


BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
SELECT_FIELDS = (
    "unique_key,created_date,closed_date,agency,complaint_type,"
    "descriptor,borough,latitude,longitude,incident_zip,status"
)
#数据来源以及我进行可视化时候所需要用到的变量（特征），防止下载所有的数据

CVD_SAFE = px.colors.qualitative.Safe
# ColorBrewer Safe (color-vision-friendly categorical palette)友好分类色板，用于时间趋势图


OKABE_ITO = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]
# Okabe-Ito (color-vision-friendly agency palette)用于给网络中的节点着色


VIRIDIS_RGB = [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37],
]
# Viridis-like continuous color scale (for the Pydeck heatmap)用于地图热力图

WEEKDAY_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#用于热力图展示星期名称

# ---------------------------------------------------------------------------
# 获取数据和清洗数据
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching 311 data from NYC Open Data…")
def load_311_data(start_iso: str, end_iso: str, max_rows: int | None = None,) -> pd.DataFrame:
    """Fetch 311 requests for a date range from the Socrata API and perform basic cleaning."""
    rows: list[dict] = []
    #存放从API获取的记录
    page_size = 50_000
    #每页请求
    offset = 0
    #每页起始位置
    where = f"created_date between '{start_iso}T00:00:00' and '{end_iso}T23:59:59'"
    #mask：筛选指定日期范围内创建的请求

    while max_rows is None or len(rows) < max_rows:#若获取的数据没有达到max就接着获取
        limit = page_size if max_rows is None else min(page_size, max_rows - len(rows))
        params = {
            "$where": where,
            "$select": SELECT_FIELDS,
            "$order": "created_date DESC",
            "$limit": limit,
            "$offset": offset,
        }
        #建立API查询参数
        resp = requests.get(BASE_URL, params=params, timeout=120)
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        rows.extend(chunk)
        offset += len(chunk)
        if len(chunk) < limit:
            break
    #获取数据时用于判断是否出现错误
    df = pd.DataFrame(rows if max_rows is None else rows[:max_rows])
    if df.empty:
        return df

    # Deduplicate and convert data types
    df = df.drop_duplicates(subset=["unique_key"])#根据unique key进行去重
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
    #转换数据为日期时间格式
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    #把经纬度转换成数值

    # Derive time features for the time series and calendar heatmap
    df["date"] = df["created_date"].dt.date
    df["month_label"] = df["created_date"].dt.strftime("%Y-%m")
    df["weekday_idx"] = df["created_date"].dt.weekday
    #根据时间派生出时间特征（日期、年月、星期编号）
    df["weekday_name"] = df["weekday_idx"].map(lambda i: WEEKDAY_EN[i] if pd.notna(i) else None)
    #改成对应的星期名称
    # Normalize borough text (trim whitespace)
    df["borough"] = df["borough"].astype(str).str.strip() #改成字符串类型，并清除首尾空格
    df.loc[df["borough"].isin(["", "nan", "None"]), "borough"] = "Unspecified"#统一标记

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------
def complaint_agency_map(df: pd.DataFrame) -> dict[str, str]:#记录每种投诉类型最常对应的响应机构
    """Map each complaint type to its most common responding agency, for node colors."""
    sub = df.dropna(subset=["complaint_type", "agency"])#移除投诉类型和机构缺失的数据
    if sub.empty:
        return {}
    return (
        sub.groupby("complaint_type")["agency"] #按complaint_type分类，并选取每组的机构字段
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")#对每种投诉类型，取出现次数最多的机构，
        .to_dict()
    )


def build_cooccurrence_graph(df: pd.DataFrame, min_count: int) -> tuple[nx.Graph, pd.DataFrame]:
    #返回网络图和共现次数表
    """Build an undirected weighted bipartite graph of complaint types and descriptors."""
    sub = df.dropna(subset=["complaint_type", "descriptor"])#删除投诉类型和描述词为空的记录
    if sub.empty:
        return nx.Graph(), pd.DataFrame(columns=["complaint_type", "descriptor", "weight"])
    pairs = (
        sub.groupby(["complaint_type", "descriptor"])
        .size()
        .reset_index(name="weight")
    )
    #按投诉类型和描述词的组合分组，统计每种组合出现的次数，次数保存在 weight 列
    pairs = pairs[pairs["weight"] >= min_count]
    #保留>mincount的组合
    G = nx.Graph()
    for _, r in pairs.iterrows():
        G.add_edge(
            f"Complaint type · {r['complaint_type']}",
            f"Descriptor · {r['descriptor']}",
            weight=float(r["weight"]),
        )
        #建立一条连接两个节点的边：一端是投诉类型；另一端是描述词；边的 weight 是这组组合出现的次数。
    return G, pairs


def make_network_figure( 
    G: nx.Graph,
    agency_map: dict[str, str],
    size_mode: str,
    keyword: str,
    max_nodes: int = 300,
) -> go.Figure | None:
    #绘制网络图
    """Draw an interactive Plotly network; keep the highest-degree nodes if there are too many."""
    if G.number_of_nodes() == 0:
        return None
    if G.number_of_nodes() > max_nodes:
        keep = [n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]]
        G = G.subgraph(keep).copy()
    #若大于maxnode就保留连接数最高的300个节点
    pos = nx.spring_layout(G, k=0.45, seed=42, weight="weight", iterations=80)
    #用弹簧布局计算节点坐标。边权重会影响布局；seed=42 让结果可重复；iterations=80 设置布局计算的迭代次数。
    
    #对应布局里面的size_mode的选择，选择不同的指标作为节点大小
    if size_mode == "Weighted degree":#加权度
        metric = dict(G.degree(weight="weight"))
    elif size_mode == "Betweenness centrality":#介数中心性
        metric = nx.betweenness_centrality(G, weight="weight")
    else:
        metric = dict(G.degree()) #普通度数
    #------------------------------------------------------------------------------------------
    vals = np.array(list(metric.values()), dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max()) if vals.max() > 0 else 1.0

    def scale(v: float) -> float: #内部函数，把指标值转成图上的节点尺寸
        if vmax <= vmin:
            return 15.0
        return 8.0 + 22.0 * (v - vmin) / (vmax - vmin)
    #指标越大，节点越大（指标映射到8-30尺寸范围）
    #——————————————————————————————————————————————————————————————————————————————————————
    # Edges
    ex, ey = [], []
    for u, v in G.edges(): #遍历网格中的边
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        #取出边两端节点的坐标
        ex += [x0, x1, None]
        ey += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=0.6, color="#d0d0d0"),
            hoverinfo="skip", showlegend=False,
        )
    )
    #把所有边画成浅灰色线条。hoverinfo="skip" 表示鼠标移到边上时不显示信息；showlegend=False 表示边不占用图例项目
    type_nodes = [n for n in G.nodes() if n.startswith("Complaint type")]
    desc_nodes = [n for n in G.nodes() if n.startswith("Descriptor")]
    #根据节点前缀，把节点分成投诉类型和描述词两类。
    
    # Complaint type nodes: color by agency
    agencies = sorted({agency_map.get(n.replace("Complaint type · ", ""), "Unknown") for n in type_nodes})
    for i, ag in enumerate(agencies):
        nodes = [n for n in type_nodes if agency_map.get(n.replace("Complaint type · ", ""), "Unknown") == ag]
        #筛选出属于当前机构的投诉类型节点。
        if not nodes:
            continue
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                mode="markers+text",
                marker=dict(
                    size=[scale(metric[n]) for n in nodes],
                    color=OKABE_ITO[i % len(OKABE_ITO)],
                    line=dict(width=1, color="#333333"),
                ),
                text=[n.replace("Complaint type · ", "") for n in nodes],
                textposition="top center",
                textfont=dict(size=9, color="#222222"),
                name=f"Agency: {ag}",
                hoverinfo="text",
                hovertext=[f"{n}<br>{size_mode} = {metric[n]:.2f}" for n in nodes],
            )
        )
        #在图上添加当前机构负责的投诉类型节点：
        #- x、y：节点位置；
        #- mode="markers+text"：显示圆点和节点文字；
        #- size：按选定指标调整大小；
        #- color：从 Okabe–Ito 色板选择颜色；
        #- text：显示投诉类型名称；
        #- name：设置图例里的机构名称；
        #- hovertext：鼠标悬停时显示节点和指标值。

        

    # Descriptor nodes: gray by default; highlight keyword matches
    highlight = [n for n in desc_nodes if keyword and keyword.lower() in n.lower()]
    #找出名称中包含关键词的描述词节点
    normal = [n for n in desc_nodes if n not in highlight]
    #剩余描述词节点作为普通节点
    if normal:
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in normal], y=[pos[n][1] for n in normal],
                mode="markers",
                marker=dict(
                    size=[scale(metric[n]) for n in normal],
                    color="#BBBBBB",
                    line=dict(width=1, color="#666666"),
                ),
                name="Descriptors",
                hoverinfo="text",
                hovertext=[f"{n}<br>{size_mode} = {metric[n]:.2f}" for n in normal],
            )
        )
    if highlight:
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in highlight], y=[pos[n][1] for n in highlight],
                mode="markers",
                marker=dict(
                    size=[scale(metric[n]) + 5 for n in highlight],
                    color="#F0E442",
                    line=dict(width=3, color="#B8860B"),
                ),
                name=f"Keyword match: {keyword}",
                hoverinfo="text",
                hovertext=[f"{n}<br>{size_mode} = {metric[n]:.2f}" for n in highlight],
            )
        )
    
    #普通描述词：灰色节点；匹配关键词的描述词：黄色节点，尺寸更大，边框更粗。
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=650,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    #设置网络图图例方向、边距、背景、图高，并隐藏坐标轴。网络图的坐标只用于定位节点，没有实际业务含义。
    return fig


# ---------------------------------------------------------------------------
# Tab 1 · Map
# ---------------------------------------------------------------------------
def map_tab(df: pd.DataFrame) -> None:
    st.subheader("🗺 Geographic Distribution of 311 Requests")
    st.caption(
        "Brighter colors indicate a higher density of requests. "
        "The heatmap uses latitude and longitude and includes only records with valid coordinates."
    )

    top_types = df["complaint_type"].value_counts().head(30).index.tolist()#最常见的前30种投诉类型
    choices = ["All"] + top_types
    ctype = st.selectbox("Complaint type (Top 30)", choices, key="map_ctype")

    sub = df.copy()
    if ctype != "All":
        sub = sub[sub["complaint_type"] == ctype]
    #如果用户选择了某一种投诉类型，就只保留这一类记录。
    geo = sub.dropna(subset=["latitude", "longitude"])#移除缺少经纬度的记录，形成地图数据。

    # Text summary (equivalent to alt text)
    if len(sub) == 0:
        st.warning("No records match the current filters.")
        return
    st.info(
        f"Filtered results: {len(sub):,} requests, including {len(geo):,} with valid coordinates. "
        f"The current view covers {geo['borough'].nunique()} boroughs."
    )

    if len(geo) == 0:
        st.warning("No valid coordinates are available, so the map cannot be drawn.")
        return

    layer = pdk.Layer(
        "HeatmapLayer",
        data=geo,
        get_position=["longitude", "latitude"],
        radius=180,
        color_range=VIRIDIS_RGB,
        opacity=0.7,
    )
    #建立 Pydeck 热力图层：
    #- data=geo：使用有效坐标记录；
    #- get_position：指定经度和纬度列；
   # - radius：控制每个点周围的扩散范围；
    #- color_range：使用前面定义的色带；
    #- opacity：设置透明度。
    
    
    view_state = pdk.ViewState(
        latitude=float(geo["latitude"].mean()),
        longitude=float(geo["longitude"].mean()),
        zoom=10,
        pitch=30,
    )
    #将地图中心设为当前记录的平均经纬度，设置缩放级别和倾斜角度。
    
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip={"text": "{borough}\n{complaint_type}"},
        ),
        use_container_width=True,
    )
    #组装地图并显示在 Streamlit 页面中
    st.caption(
        "Accessibility: The heatmap ranges from dark blue (lower density) to bright yellow (higher density). "
        "The viridis scale is designed to be accessible to people with red-green color vision deficiencies. "
        "The text summary above provides the chart's key figures."
    )


# ---------------------------------------------------------------------------
# Tab 2 · Time Series
# ---------------------------------------------------------------------------
def time_tab(df: pd.DataFrame, start_d: date, end_d: date) -> None:
    st.subheader("📈 Request Trends Over Time")
    st.caption("Request volume by complaint type; the calendar heatmap below shows density by month and weekday.")

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        granularity = st.radio("Time interval", ["Daily", "Weekly", "Monthly"], horizontal=False, index=1)
        #提供日、周两种时间粒度，默认选中 "Weekly"
    with col2:
        all_agencies = sorted(df["agency"].dropna().unique())
        default_ag = [a for a in ["NYPD", "HPD", "DSNY", "DOT"] if a in all_agencies]
        agencies = st.multiselect("Agencies", all_agencies, default=default_ag or all_agencies[:3])
        #选择机构
    with col3:
        top_n = st.slider("Number of complaint types to display", 3, 12, 6)
        #选择显示多少种投诉类型
        
        
    
    sub = df[df["agency"].isin(agencies)] if agencies else df
    #选择了机构时，仅使用这些机构的数据；如果没有选择，就使用全部数据
    if sub.empty:
        st.warning("No records match the current filters.")
        return

    freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}
    freq = freq_map[granularity]
    #将界面中的英文选项转成 Pandas 使用的重采样代码：D：按天；W-MON：以周一为周界；
    
    # Preserve the agency filter. If no agency is selected, use all agencies.
    agency_data = df[df["agency"].isin(agencies)] if agencies else df

    if agency_data.empty:
        st.warning("No records match the selected agencies.")
        return

    # Find the first and last dates with data for the selected agencies.
    available_dates = agency_data["created_date"].dropna().dt.date
    #available_dates 中包含这些机构实际有请求记录的日期。
    
    if available_dates.empty:
        st.warning("No valid request dates are available for the selected agencies.")
        return

    data_start = max(start_d, min(available_dates))
    data_end = min(end_d, max(available_dates))

    selected_range = st.date_input(
        "Time Pulse date range",
        value=(data_start, data_end),
        min_value=start_d,
        max_value=end_d,
        key="time_pulse_date_range",
        help=(
            "The default range spans the first through last request dates available "
            "for the selected agencies. You can choose any subrange within the global date range."
            ),
        )
    #创建日期选择控件，并把用户选择的结果存入 selected_range

    if isinstance(selected_range, (tuple, list)):
        if len(selected_range) == 2:
            pulse_start, pulse_end = selected_range
        elif len(selected_range) == 1:
            pulse_start = pulse_end = selected_range[0]
        else:
            st.info("Select a start and end date to display the time series.")
            return
    else:
        pulse_start = pulse_end = selected_range
    #选择时间的三种情况

    if pulse_start > pulse_end:
        st.warning("The start date must be on or before the end date.")
        return

    # Apply the selected date range first.
    date_mask = df["created_date"].dt.date.between(pulse_start, pulse_end)
    
    sub = df.loc[date_mask].copy()
    #只保留日期范围内的记录，并复制成新的 DataFrame sub
    
    # Preserve the agency filter. If no agency is selected, use all agencies.
    if agencies:
        sub = sub[sub["agency"].isin(agencies)]

    if sub.empty:
        st.warning("No records match the selected date range and agencies.")
        return

    top_types = sub["complaint_type"].value_counts().head(top_n).index.tolist()
    ts_input = sub[sub["complaint_type"].isin(top_types)]
    #找出机构筛选后的前 top_n 种投诉类型，并保留
    
    ts = (
        ts_input.set_index("created_date")
        .groupby("complaint_type")["unique_key"]
        .resample(freq)
        .count()
        .reset_index(name="count")
    )
    #按投诉类型和时间段统计请求数：
    #1. 把创建时间设为索引；
    #2. 按投诉类型分组；
    #3. 按所选时间粒度重新分组；
    #4. 计数；
    #5. 把结果转回普通表格，并将计数列命名为 count。

    fig = px.line(
        ts, x="created_date", y="count", color="complaint_type",
        color_discrete_sequence=CVD_SAFE,
        labels={"created_date": "Date", "count": "Number of requests", "complaint_type": "Complaint type"},
    )
    #绘制折线图。每种投诉类型对应一条线，横轴是日期，纵轴是请求数量。
    fig.update_layout(
        legend_title_text="Complaint type",
        hovermode="x unified",
        height=420,
        plot_bgcolor="#ffffff",
    )
    #折线图布局
  
    fig.update_xaxes(
        type="date",
        range=[pulse_start.isoformat(), pulse_end.isoformat()],
        dtick="D1",
    )
    #保持x轴在选定范围内，并使用每日日历刻度。
    
    st.plotly_chart(fig, use_container_width=True)

    # Text summary
    if len(ts):
        peak = ts.loc[ts["count"].idxmax()]
        st.info(
            f"At the selected interval, the peak occurred on {pd.to_datetime(peak['created_date']).date()} "
            f"for “{peak['complaint_type']},” with {int(peak['count'])} requests."
        )
        #显示峰值所在日期、投诉类型以及请求数量。
    
    # Calendar heatmap
    st.subheader("Request Density by Month and Weekday")
    cal = (
        sub.groupby(["month_label", "weekday_name"])
        .size()
        .reset_index(name="count")
    )
    #对筛选后的 sub 数据，按月份和星期名称组合分组，并统计每组请求数量。
    if not cal.empty:
        # Keep weekdays in calendar order
        cal["weekday_name"] = pd.Categorical(cal["weekday_name"], categories=WEEKDAY_EN, ordered=True)
        cal["month_label"] = pd.Categorical(cal["month_label"],categories=sorted(cal["month_label"].unique()),ordered=True)
        #把英文星期名称设置成有序分类变量，顺序为周一到周日。
        cal = cal.sort_values("weekday_name")
        cal_fig = px.density_heatmap(
            cal, x="month_label", y="weekday_name", z="count",
            histfunc="sum",
            color_continuous_scale="Viridis",
            labels={"month_label": "Month", "weekday_name": "Weekday", "count": "Number of requests"},
        )
        #绘制热力图
        cal_fig.update_layout(height=320, plot_bgcolor="#ffffff",xaxis_type="category",)
        #设置热力图高度和背景颜色。
        
        st.plotly_chart(cal_fig, use_container_width=True)
        busiest = cal.loc[cal["count"].idxmax()]
        st.info(
            f"The busiest month-weekday combination is {busiest['weekday_name']} in {busiest['month_label']}, "
            f"with {int(busiest['count'])} requests."
        )


# ---------------------------------------------------------------------------
# Tab 3 · Language + Network
# ---------------------------------------------------------------------------
def network_tab(df: pd.DataFrame) -> None:
    st.subheader("🔗 Complaint Type and Descriptor Co-occurrence Network")
    st.caption(
        "Nodes represent complaint types or descriptors; edges connect pairs that occur in the same records. "
        "Thicker edges indicate more frequent co-occurrence. The bar chart below provides a text-based view of descriptor frequency."
    )

    if df["descriptor"].dropna().empty:
        st.warning("No descriptor values are available in the current data, so a network cannot be built.")
        return

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        threshold = st.slider("Minimum co-occurrence count", 5, 500, 30, step=5)#最低出现次数阈值
    with col2:
        size_mode = st.selectbox("Node size metric", ["Weighted degree", "Degree", "Betweenness centrality"])
    with col3:
        keyword = st.text_input("Highlight descriptor keyword (optional)", "")
        #创建关键词输入框。输入文字后，名称中包含该关键词的描述词节点会被高亮；不输入时则不高亮任何关键词。

    G, pairs = build_cooccurrence_graph(df, threshold)
    agency_map = complaint_agency_map(df)
    #找出每种投诉类型最常对应的响应机构。

    if G.number_of_nodes() == 0:
        st.warning("No co-occurrence edges meet this threshold. Try lowering it.")
        return

    # Text summary (equivalent to alt text)
    weighted_deg = sorted(G.degree(weight="weight"), key=lambda x: x[1], reverse=True)
    top_node, top_val = weighted_deg[0]
    st.info(
        f"Network size: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. "
        f"The most strongly connected node is “{top_node}” (weighted degree: {top_val:.0f}). "
        f"Network density: {nx.density(G):.4f}."
    )

    fig = make_network_figure(G, agency_map, size_mode, keyword)
    
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("The network graph could not be rendered.")
    #绘制图

    st.caption(
        "Accessibility: Complaint type nodes are colored by responding agency using the color-vision-friendly Okabe-Ito palette. "
        "Descriptor nodes are gray, and keyword matches are highlighted in yellow. "
        "If the interactive graph is difficult to read, use the frequency chart below."
    )

    # Text-based view
    st.subheader("Top 15 Descriptors by Frequency")
    top_desc = (
        df["descriptor"].dropna().value_counts().head(15)
        .rename_axis("Descriptor").reset_index(name="count")
    )
    #计算描述词频次：
    if not top_desc.empty:
        bar = px.bar(
            top_desc, x="count", y="Descriptor", orientation="h",
            color_discrete_sequence=["#0072B2"],
            labels={"count": "Occurrences", "Descriptor": ""},
        )
        bar.update_layout(height=420, yaxis=dict(autorange="reversed"), plot_bgcolor="#ffffff")
        st.plotly_chart(bar, use_container_width=True)
        st.info(
            f"The most frequent descriptor is “{top_desc.iloc[0]['Descriptor']},” "
            f"appearing {int(top_desc.iloc[0]['count'])} times."
        )
        #创建横向条形图：横轴是出现次数；纵轴是描述词；orientation="h" 表示横向；使用蓝色；将横轴标签设置为 Occurrences，纵轴不显示额外标题。

# ---------------------------------------------------------------------------
# Tab 4 · Data Story
# ---------------------------------------------------------------------------
def compute_kpis(df: pd.DataFrame, G: nx.Graph) -> dict:#计算汇总指标
    total = len(df)#数据总和
    closed = df.dropna(subset=["closed_date"]).copy()#记录有关闭时间的数据
    if len(closed):
        med_days = (
            (closed["closed_date"] - closed["created_date"]).dt.total_seconds() / 86400
        ).median()
        #计算每条请求从创建到关闭的时长的中位数， 一天86400秒
    else:
        med_days = float("nan")
    top_type = df["complaint_type"].value_counts().idxmax() if len(df) else "—"
    #找出出现次数最多的投诉类型
    density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
    #计算网络密度
    top_borough = df["borough"].value_counts().idxmax() if len(df) else "—"
    #找出请求最多的borough
    return {
        "total": total,
        "med_days": med_days,
        "top_type": top_type,
        "density": density,
        "top_borough": top_borough,
    }


def story_tab(df: pd.DataFrame, G: nx.Graph) -> None:
    st.subheader("📖 Data Story")
    st.caption("This page summarizes key findings from the first three views and presents the main takeaways in text.")

    kpi = compute_kpis(df, G)
    #里面有total, med_day, top_type,density, top_borough
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total requests", f"{kpi['total']:,}")
    c2.metric(
        "Median time to close",
        "—" if pd.isna(kpi["med_days"]) else f"{kpi['med_days']:.1f} days",
    )
    c3.metric("Most frequent complaint type", str(kpi["top_type"]))
    c4.metric("Network density", f"{kpi['density']:.4f}")
    #数据展示
    
    st.markdown(
        f"""
### 1. Geographic Distribution
During the selected time window, **{kpi['top_borough']}** had the most requests.
The map tab shows where requests are concentrated across the city.
**Request density reflects reporting behavior, not the objective density of underlying problems.**
The dataset includes service requests routed to specific agencies and excludes requests handled by organizations such as NYCHA and DOC through separate customer service systems.

### 2. Temporal Patterns
The time series tab shows weekly patterns and month-to-month variation in request volume.
The calendar heatmap locates these patterns by month and weekday.
Seasonal events (such as heating season or extreme weather) and reporting behavior may both influence these patterns, so they should not be read directly as changes in problem severity.

### 3. Language Patterns
The network tab visualizes co-occurrence between `complaint_type` and `descriptor` as a bipartite graph.
Some complaint types may be strongly associated with a small set of descriptors, while others may have a more varied descriptor distribution.
These differences may reflect how consistently residents describe different issues and suggest that descriptor sparsity should be considered when building text classification or topic modeling systems.
"""
    )

    st.markdown("---")
    st.markdown(
        """
### Data Limitations (from the data dictionary)
- The dataset excludes inquiries and requests handled by organizations such as NYCHA and DOC through separate customer service systems.
- It reflects requests submitted by residents, not an objective snapshot of all service conditions in the city.
- `Problem Detail` (Descriptor) is not present in every record, so network and text analyses cover only records with a descriptor.
- Location fields are geocoded, and their completeness cannot be fully verified.
"""
    )


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="NYC 311 Signals · Multimodal Exploration",
        page_icon="🗽",
        layout="wide",
    )

    st.title("🗽 NYC 311 Signals — Multimodal Exploration")
    st.caption(
        "Explore NYC 311 requests through a map, time series, language and network views, and a data story."
    )

    today = date.today()

    # ---------------- Sidebar: choose retrieval settings ----------------
    with st.sidebar:
        st.header("Data Retrieval")
        download_mode = st.radio(
            "Data retrieval mode",
            ["Prioritize date range", "Limit number of records"],
            help=(
                "Choose a date range to retrieve all matching requests, or set a maximum "
                "number of records to retrieve."
            ),
        )

        if download_mode == "Prioritize date range":
            selected_dates = st.date_input(
                "Request creation date range",
                value=(today - timedelta(days=30), today - timedelta(days=1)),
                min_value=date(2020, 1, 1),
                max_value=today,
                help=(
                    "The app will retrieve all matching requests in this date range. "
                    "Large date ranges may take longer."
                ),
            )
            max_rows = None

            if isinstance(selected_dates, (tuple, list)):
                if len(selected_dates) == 2:
                    start_d, end_d = selected_dates
                elif len(selected_dates) == 1:
                    start_d, end_d = selected_dates[0], None
                else:
                    start_d, end_d = None, None
            else:
                start_d = end_d = selected_dates

            if start_d is None or end_d is None:
                st.info("Select both dates, then click **Load data**.")

        else:
            max_rows = st.slider(
                "Maximum number of records",
                min_value=20_000,
                max_value=200_000,
                value=100_000,
                step=20_000,
                help="The app retrieves up to this many most recent requests.",
            )
            # Query the dataset's supported period; the API returns newest requests first.
            start_d, end_d = today - timedelta(days=30), today
            st.caption(
                "After loading, the app will show the actual date coverage of the records retrieved."
            )

        load_clicked = st.button(
            "Load data",
            type="primary",
            use_container_width=True,
        )

    # A signature lets the app detect changed settings without loading automatically.
    retrieval_signature = (download_mode, start_d, end_d, max_rows)

    if load_clicked:
        if start_d is None or end_d is None:
            st.warning("Select a complete date range before loading data.")
            st.stop()
        if start_d > end_d:
            st.error("The start date must be on or before the end date.")
            st.stop()

        try:
            with st.spinner(
                "Fetching requests from NYC Open Data. This may take a while…"
            ):
                loaded_df = load_311_data(
                    str(start_d),
                    str(end_d),
                    max_rows=max_rows,
                )
            st.session_state["nyc311_loaded_df"] = loaded_df
            st.session_state["nyc311_retrieval_signature"] = retrieval_signature
            st.session_state["nyc311_retrieval_mode"] = download_mode
            st.session_state["nyc311_max_rows"] = max_rows
        except requests.RequestException as exc:
            st.error(
                "Could not retrieve data. Check your internet connection or try again later.\n\n"
                f"Technical details: {exc}"
            )
            st.stop()

    if "nyc311_loaded_df" not in st.session_state:
        st.info(
            "Choose a retrieval mode and its settings in the sidebar, then click **Load data**."
        )
        return

    if st.session_state.get("nyc311_retrieval_signature") != retrieval_signature:
        st.info(
            "Your retrieval settings have changed. Click **Load data** to fetch data for these settings."
        )
        return

    df = st.session_state["nyc311_loaded_df"].copy()
    if df.empty:
        st.error("No data was returned. Try a different date range or retrieval setting.")
        return

    # Report the date coverage of the records actually retrieved, before borough filtering.
    valid_dates = df["created_date"].dropna().dt.date
    if valid_dates.empty:
        st.error("The retrieved records do not contain valid creation dates.")
        return
    loaded_start = min(valid_dates)
    loaded_end = max(valid_dates)

    # ---------------- Sidebar: borough filter ----------------
    with st.sidebar:
        st.divider()
        all_boroughs = sorted(df["borough"].dropna().unique())
        sel_boroughs = st.multiselect(
            "Borough",
            all_boroughs,
            default=all_boroughs,
            help="Uncheck a borough to exclude it.",
        )

    if not sel_boroughs:
        st.warning("Select at least one borough.")
        return
    df = df[df["borough"].isin(sel_boroughs)].reset_index(drop=True)

    st.success(
        f"Loaded **{len(df):,}** requests. Actual date coverage: "
        f"**{loaded_start} → {loaded_end}**."
    )
    if st.session_state["nyc311_retrieval_mode"] == "Number of records":
        st.caption(
            f"Record limit: {st.session_state['nyc311_max_rows']:,}. "
            "The date coverage above reflects the records actually retrieved."
        )

    # Download the currently loaded and borough-filtered data as an Excel workbook.
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="311 Requests")

    st.download_button(
        label="Download data as Excel",
        data=excel_buffer.getvalue(),
        file_name=f"nyc311_requests_{loaded_start}_{loaded_end}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("About the Dataset and Variables", expanded=False):
        st.markdown(
            """
### Dataset Overview

This project uses the **311 Service Requests from 2020 to Present** dataset,
published by NYC Open Data. Each row represents a service request submitted
through New York City's 311 system.

- **Source:** [NYC Open Data — 311 Service Requests](https://data.cityofnewyork.us/resource/erm2-nwe9.json)
- **Socrata dataset ID:** `erm2-nwe9`
- **Important limitation:** These records reflect requests submitted through 311. They do not represent every service issue or an objective measure of problem severity across the city.

### Variables Used

| Variable | Description | Used for |
|---|---|---|
| `unique_key` | Unique identifier for a service request | Counting requests and removing duplicates |
| `created_date` | Date and time when the request was created | Date filtering and time trend analysis |
| `closed_date` | Date and time when the request was closed | Calculating median time to close |
| `agency` | City agency responsible for responding | Agency filtering and network node colors |
| `complaint_type` | General category of the request | Complaint type comparisons and network nodes |
| `descriptor` | More specific description of the reported issue | Descriptor frequency and co-occurrence network |
| `borough` | Borough associated with the request | Geographic filtering and summaries |
| `latitude` / `longitude` | Coordinates associated with the request | Mapping |
| `incident_zip` | ZIP code associated with the incident | Retrieved, but not currently used in visualizations |
| `status` | Current status of the request | Retrieved, but not currently used in visualizations |

The app also derives `date`, `month_label`, `weekday_idx`, and `weekday_name` from `created_date` for time-based views.
"""
        )

    st.info(
        f"Requests displayed: **{len(df):,}** · Actual date coverage: "
        f"**{loaded_start} → {loaded_end}** · Boroughs: {', '.join(sel_boroughs)}"
    )

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Map Explorer", "📈 Time Pulse", "🔗 Language & Network", "📖 Data Story"]
    )

    with tab1:
        map_tab(df)
    with tab2:
        time_tab(df, loaded_start, loaded_end)
    with tab3:
        network_tab(df)
    with tab4:
        G_story, _ = build_cooccurrence_graph(df, 30)
        story_tab(df, G_story)


if __name__ == "__main__":
    main()
