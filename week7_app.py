"""
Week 7 Text Data — Streamlit assignment app

Corpus: selected categories from scikit-learn's 20 Newsgroups training set.
The app provides:
  1) honest sorted term-frequency bars,
  2) BERTopic topic map + selectable topic terms + optional hierarchy,
  3) a UMAP document projection colored by source-category metadata,
  4) an optional raw-count vs mean-TF-IDF comparison.

First run needs internet access to download 20 Newsgroups and the
sentence-transformer model used by BERTopic.

Install dependencies:
  pip install streamlit scikit-learn bertopic sentence-transformers umap-learn plotly pandas numpy

Run:
  streamlit run week07_text_streamlit.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import umap
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


st.set_page_config(
    page_title="Text Data Explorer",
    page_icon="📝",
    layout="wide",
)#设置浏览器标签页标题

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#指定句向量模型的名称。BERTopic 和 UMAP 都会依赖这个模型生成的文档向量。

DEFAULT_CATEGORIES = (
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
)
#4个新闻组类别

st.title("Text Data Explorer")
st.caption(
    "DATS 6401 · Week 7 · 20 Newsgroups — frequency, BERTopic, and document embeddings"
)
st.markdown(
    """
This app uses four 20 Newsgroups categories as **source metadata**. It removes
headers, signatures, and quoted text before analysis. The corpus is sampled
equally by category so raw category counts are not driven simply by different
sample sizes.
"""
)



@st.cache_data(show_spinner="Loading the 20 Newsgroups corpus…")
#这是 Streamlit 的数据缓存装饰器
def load_corpus(categories: tuple[str, ...], per_category: int) -> pd.DataFrame:
    """Download/cache a reproducible, balanced subset of 20 Newsgroups."""
    bunch = fetch_20newsgroups(
        subset="train",
        categories=list(categories),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
        download_if_missing=True,
    )
    #读取数据
    
    source_names = [bunch.target_names[int(label)] for label in bunch.target]
    #把编号转换成类别名称
    frame = pd.DataFrame({"text": bunch.data, "source": source_names})
    #创建dataframe，text+source
    frame["text"] = frame["text"].map(
        lambda value: re.sub(r"\s+", " ", str(value)).strip()
    )
    #清理文本
    frame = frame[frame["text"].str.len() >= 80].copy()
    #删除字符较少的文档
    sampled_groups = []
    for _, group in frame.groupby("source", sort=True):
        sampled_groups.append(
            group.sample(
                n=min(per_category, len(group)),
                random_state=42,
            )
        )
        #从每个类别随机抽样并存储在sampled_group
        
    if not sampled_groups:
        return pd.DataFrame(columns=["text", "source"])

    return pd.concat(sampled_groups, ignore_index=True)
    #把各类别的样本拼成一张表。ignore_index=True 会重新生成从 0 开始的行索引

@st.cache_data(show_spinner=False)#缓存词频分析结果
def calculate_frequency_views(
    texts: tuple[str, ...],
    sources: tuple[str, ...],
    selected_source: str,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return corpus counts, source counts, and mean source TF-IDF."""
    count_vectorizer = CountVectorizer(
        stop_words="english", #删除常见英文常用词
        min_df=2, #至少出现两篇文档的词
        max_df=0.95,#删除出现在超过95%文档的词
        max_features=8000,
    )
    #生成词频矩阵
    count_matrix = count_vectorizer.fit_transform(texts)
    #词汇表-行代表文档，列代表词语，值表示出现次数
    count_terms = count_vectorizer.get_feature_names_out()
    #矩阵每列对应的词语
    total_counts = np.asarray(count_matrix.sum(axis=0)).ravel()
    #每个词在语料的总次数
    corpus_top = (
        pd.DataFrame({"term": count_terms, "value": total_counts})
        .sort_values("value", ascending=False)
        .head(top_n)
        .sort_values("value", ascending=True)
    )
    #最后的倒序是为了让最长条放在上面

    source_mask = np.asarray(sources) == selected_source
    #文档属于所选类别时为true
    source_counts = np.asarray(count_matrix[source_mask].sum(axis=0)).ravel()
    #先选出该类别对应的文档行，再对每个词的词频加总
    source_top = (
        pd.DataFrame({"term": count_terms, "value": source_counts})
        .sort_values("value", ascending=False)
        .head(top_n)
        .sort_values("value", ascending=True)
    )

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_df=0.95,
        max_features=8000,
    )
    #输出TF-IDF权重
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()
    #矩阵对应的词语
    mean_tfidf = np.asarray(tfidf_matrix[source_mask].mean(axis=0)).ravel()
    #求平均
    tfidf_top = (
        pd.DataFrame({"term": tfidf_terms, "value": mean_tfidf})
        .sort_values("value", ascending=False)
        .head(top_n)
        .sort_values("value", ascending=True)
    )

    return corpus_top, source_top, tfidf_top


@st.cache_resource(show_spinner="Loading the sentence embedding model…")
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load one shared sentence-transformer model."""
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Fitting BERTopic and encoding documents…")
def fit_topic_model(
    docs: tuple[str, ...],
    min_topic_size: int,
) -> tuple[BERTopic, np.ndarray, np.ndarray]:
    """Fit BERTopic and return the model, topic labels, and document embeddings."""
    embedding_model = load_embedding_model(MODEL_NAME)
    #获取句向量模型
    embeddings = embedding_model.encode(
        list(docs),
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    #文档编码成向量
    model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    #创建模型
    topics, _ = model.fit_transform(list(docs), embeddings=embeddings)
    #topics主题编号、_表示与主题概率有关
    return model, np.asarray(topics), np.asarray(embeddings)


@st.cache_data(show_spinner="Projecting document embeddings with UMAP…")
def project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Create a reproducible two-dimensional UMAP projection."""
    neighbor_count = min(15, max(2, len(embeddings) - 1))
    #计算n_neighbors参数
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbor_count,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    #创建UMAP降维器-n_components投影的维度、min_dist控制投影点之后之间可以靠得多近
    return reducer.fit_transform(embeddings)


def bar_figure(data: pd.DataFrame, title: str, x_label: str) -> go.Figure:
    """Build an ordered horizontal bar chart."""
    figure = px.bar(
        data,
        x="value",
        y="term",
        orientation="h",
        title=title,
        labels={"value": x_label, "term": "Term"},
        color_discrete_sequence=["#2E6E8E"],
    )
    #水平条形图，横轴是数值，纵轴是词语
    figure.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
    )
    return figure


with st.sidebar:
    st.header("Corpus and model settings")
    #侧边栏名称
    selected_categories = st.multiselect(
        "Choose source categories",
        options=DEFAULT_CATEGORIES,
        default=list(DEFAULT_CATEGORIES),
        help="These category labels are metadata used to color the document projection.",
    )
    #类别选择（至少两个），对应的会作为数据来源
    docs_per_category = st.slider(
        "Maximum documents per category",
        min_value=30,
        max_value=150,
        value=60,
        step=10,
        help="The app samples the same maximum number from each category.",
    )
    #每个类别抽样数
    min_topic_size = st.slider(
        "BERTopic minimum topic size",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
    )
    #BERTopic的最小主题数
    top_n = st.slider(
        "Terms shown in frequency charts",
        min_value=10,
        max_value=30,
        value=15,
        step=5,
    )
    #设置词频图显示多少词

if len(selected_categories) < 2:
    st.warning("Select at least two source categories so the metadata comparison is useful.")
    st.stop()
#防止类别只选择一个

try:
    with st.spinner("Loading and cleaning the selected documents…"):
        corpus = load_corpus(tuple(selected_categories), docs_per_category)
except Exception as error:
    st.error(
        "Could not load 20 Newsgroups. The first run needs an internet connection "
        "to download the dataset."
    )
    st.exception(error)
    st.stop()
#加载语料

if corpus.empty or len(corpus) < 10:
    st.error("The selected categories did not produce enough non-empty documents.")
    st.stop()
#判断是否少于10篇，不足以主题模型

docs = tuple(corpus["text"].tolist())
sources = tuple(corpus["source"].tolist())

st.sidebar.success(f"{len(corpus)} documents · {len(selected_categories)} categories")
#侧边栏显示加载到的文档总数和类别数

st.write(
    f"**Corpus:** {len(corpus)} documents from "
    f"{len(selected_categories)} newsgroup categories."
)

st.dataframe(
    corpus.groupby("source", as_index=False)
    .size()
    .rename(columns={"source": "Source metadata", "size": "Documents"}),
    hide_index=True,
    use_container_width=True,
)
#主页展示

try:
    topic_model, topic_labels, embeddings = fit_topic_model(docs, min_topic_size)
    coordinates = project_embeddings(embeddings)
except Exception as error:
    st.error(
        "BERTopic or the embedding projection could not run. Check the installation "
        "instructions and make sure the first model download can access the internet."
    )
    st.exception(error)
    st.stop()
#训练BERTopic， 再把embedding向量利用UMAP投影到二维中

results = corpus.copy()
results["topic_id"] = topic_labels
results["topic_label"] = results["topic_id"].map(
    lambda value: "Outlier / unassigned (-1)"
    if int(value) == -1
    else f"Topic {int(value)}"
)
#利用map将主题编号换成Topic0、Topic1

results["text_preview"] = results["text"].str.slice(0, 260)
#截取前260个字符
topic_info = topic_model.get_topic_info()
#从 BERTopic 获取主题汇总表，通常包括主题编号、文档数量、主题名称和代表性词语。
topic_lookup: dict[str, int | None] = {"All topics": None}
for _, row in topic_info.iterrows():
    topic_id = int(row["Topic"])
    count = int(row["Count"])
    name = str(row["Name"])
    if topic_id == -1:
        label = f"Outliers (-1) · {count} documents"
    else:
        label = f"Topic {topic_id} · {count} documents · {name}"
    topic_lookup[label] = topic_id
    #为每个主题建立显示名称和对应编号。

selected_topic_label = st.sidebar.selectbox(
    "Topic selector",
    options=list(topic_lookup.keys()),
    index=0,
    help="Filter the topic detail and document projection to one topic.",
)
selected_topic = topic_lookup[selected_topic_label]
#选择想展示的主题

if selected_topic is None:
    filtered_results = results
else:
    filtered_results = results[results["topic_id"] == selected_topic]


frequency_tab, topics_tab, embedding_tab = st.tabs(
    ["1 · Frequency and TF–IDF", "2 · BERTopic", "3 · Embedding projection"]
)

with frequency_tab:
    st.subheader("An honest frequency view")
    st.markdown(
        "The sorted bars encode raw term counts by bar length. Stop words are "
        "removed, and terms appearing in only one document are excluded."
    )

    source_for_keyness = st.selectbox(
        "Compare raw counts and TF–IDF for this source category",
        options=sorted(set(sources)),
    )

    try:
        corpus_top, source_top, tfidf_top = calculate_frequency_views(
            docs,
            sources,
            source_for_keyness,
            top_n,
        )
    except ValueError as error:
        st.error(f"Could not build the term-frequency matrix: {error}")
        st.stop()
    #取得三张结果表
    st.plotly_chart(
        bar_figure(
            corpus_top,
            "Most frequent terms across the selected corpus",
            "Raw count",
        ),
        use_container_width=True,
    )


    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            bar_figure(
                source_top,
                f"Raw counts · {source_for_keyness}",
                "Occurrences in selected source",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            bar_figure(
                tfidf_top,
                f"Mean TF–IDF · {source_for_keyness}",
                "Mean TF–IDF score",
            ),
            use_container_width=True,
        )

    st.info(
        "**What this reveals:** raw counts show what appears often; mean TF–IDF "
        "highlights terms that are frequent in the selected source but less common "
        "across the full corpus. **Possible misreading:** high raw frequency does "
        "not mean a term is distinctive, and mean TF–IDF is a descriptive weight, "
        "not a statistical significance test."
    )
    
    

with topics_tab:
    st.subheader("BERTopic topic model")
    st.markdown(
        "BERTopic embeds documents, groups them, and describes each group with "
        "distinctive terms. Topic labels are model-generated summaries, so inspect "
        "the terms and example documents before naming a topic."
    )

    valid_topic_count = int((topic_info["Topic"] != -1).sum())
    #统计普通主题的数量
    if valid_topic_count:
        st.plotly_chart(
            topic_model.visualize_topics(),
            use_container_width=True,
        )
        #plot
        st.caption(
            "The topic map is a projection of topic representations. Bubble distances "
            "are approximate and should not be interpreted as exact measurements."
        )
    else:
        st.warning("BERTopic did not identify any non-outlier topics for these settings.")

    if selected_topic is not None:
        st.markdown(f"#### Selected: {selected_topic_label}")
        topic_words = topic_model.get_topic(int(selected_topic))
        if topic_words:
            term_df = pd.DataFrame(topic_words[:12], columns=["term", "weight"])
            term_df = term_df.sort_values("weight", ascending=True)
            term_fig = px.bar(
                term_df,
                x="weight",
                y="term",
                orientation="h",
                title="Top c-TF-IDF terms for the selected topic",
                labels={"weight": "c-TF-IDF weight", "term": "Term"},
                color_discrete_sequence=["#5B8E7D"],
            )
            term_fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(term_fig, use_container_width=True)
        else:
            st.write("This topic has no available term representation.")

    st.markdown("#### Topic summary")
    st.dataframe(
        topic_info[["Topic", "Count", "Name", "Representation"]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Example documents")
    example_columns = ["source", "topic_label", "text_preview"]
    st.dataframe(
        filtered_results[example_columns].head(8),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Optional extension: hierarchical topics")
    hierarchy_signature = (
        tuple(selected_categories),
        docs_per_category,
        min_topic_size,
    )
    if st.button("Generate BERTopic hierarchy"):
        with st.spinner("Calculating the topic hierarchy…"):
            try:
                hierarchy_data = topic_model.hierarchical_topics(list(docs))
                st.session_state["hierarchy_signature"] = hierarchy_signature
                st.session_state["hierarchy_data"] = hierarchy_data
            except Exception as error:
                st.error(f"Could not calculate the hierarchy: {error}")

    if (
        st.session_state.get("hierarchy_signature") == hierarchy_signature
        and "hierarchy_data" in st.session_state
    ):
        st.plotly_chart(
            topic_model.visualize_hierarchy(
                hierarchical_topics=st.session_state["hierarchy_data"]
            ),
            use_container_width=True,
        )
        st.caption(
            "The hierarchy suggests which topics may be related or mergeable; it is "
            "an algorithmic structure, not a definitive taxonomy."
        )

    st.info(
        "**What this reveals:** recurring themes and their characteristic terms. "
        "**Possible misreading:** the number and boundaries of topics depend on the "
        "corpus, embedding model, and minimum topic size. Topic -1 contains documents "
        "the clustering step did not assign."
    )

with embedding_tab:
    st.subheader("Documents as points in a two-dimensional map")
    st.markdown(
        "Each point is one document. The default color is the document's source "
        "category metadata, rather than its BERTopic topic."
    )

    color_mode = st.radio(
        "Color points by",
        options=["Source category (metadata)", "BERTopic topic"],
        horizontal=True,
        index=0,
    )

    projection = pd.DataFrame(
        {
            "UMAP 1": coordinates[:, 0],
            "UMAP 2": coordinates[:, 1],
            "source": results["source"].to_numpy(),
            "topic_id": results["topic_id"].to_numpy(),
            "topic_label": results["topic_label"].to_numpy(),
            "text_preview": results["text_preview"].to_numpy(),
        }
    )
    if selected_topic is not None:
        projection = projection[projection["topic_id"] == selected_topic]

    color_column = (
        "source"
        if color_mode == "Source category (metadata)"
        else "topic_label"
    )
    color_title = "Source category" if color_column == "source" else "BERTopic topic"

    embedding_fig = px.scatter(
        projection,
        x="UMAP 1",
        y="UMAP 2",
        color=color_column,
        hover_data={
            "source": True,
            "topic_label": True,
            "text_preview": True,
            "UMAP 1": False,
            "UMAP 2": False,
        },
        title=f"UMAP projection · colored by {color_title.lower()}",
        color_discrete_sequence=px.colors.qualitative.Safe,
        opacity=0.78,
    )
    embedding_fig.update_traces(marker=dict(size=8))
    embedding_fig.update_layout(
        height=650,
        legend_title_text=color_title,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(embedding_fig, use_container_width=True)
    st.caption(
        "Projection: UMAP of sentence-transformer embeddings "
        f"({MODEL_NAME}), n_neighbors={min(15, max(2, len(embeddings) - 1))}, "
        "min_dist=0.1, metric=cosine, random_state=42."
    )
    st.info(
        "**What this reveals:** nearby points may use similar language or discuss "
        "related content. **Possible misreading:** UMAP compresses high-dimensional "
        "embeddings into 2D; distances, cluster sizes, and empty space are not "
        "literal semantic measurements."
    )

with st.expander("Corpus description and overall limitation"):
    st.write(
        "The 20 Newsgroups collection contains discussion posts grouped into "
        "newsgroups. This app uses a balanced sample from the selected groups and "
        "uses each group name as source metadata. The collection is historical, and "
        "the category labels are not a perfect description of every post."
    )
    st.write(
        "Text preprocessing removes headers, footers, and quoted passages, then "
        "normalizes whitespace. These choices change the vocabulary and should be "
        "reported when interpreting the charts."
    )

st.divider()
st.caption(
    "Assignment views: sorted raw frequency bars · BERTopic visualization · "
    "UMAP document projection · TF–IDF comparison and hierarchical-topic extensions"
)

