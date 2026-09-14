# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:40:44 2026

@author: ZJY
"""

# -*- coding: utf-8 -*-

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Week 3: Multivariate Data", layout="wide")

HEATMAP_INTERPRETATION = """
### Interpretation

A distinct block of positive correlations among seed size variables emerged in the heatmap. Area showed a very strong positive correlation with perimeter, 
while seed length, seed width, and groove length also exhibited strong positive correlations with both area and perimeter. 
This indicates that these variables are not entirely independent; 
they describe the overall size of wheat seeds to varying degrees and thus contain some redundant information.
 In contrast, the asymmetry coefficient showed weaker correlations with most size variables, suggesting it reflects shape characteristics different from overall dimensions.
 Correlation only captures linear relationships between variables, so while heatmaps can help reveal variable structures, they cannot independently establish causality.

"""

PCA_INTERPRETATION = """
### Interpretation

In the PC1–PC2 scatter plot, the three wheat varieties form a meaningful distribution pattern. 
The Rosa samples are primarily located on one side of PC1, the Canadian samples mainly on the opposite side, 
while most Kama samples lie between them and show some variation along the PC2 direction. 
There is still slight overlap among the three groups, so they cannot be considered completely separated;
 however, the overall distribution indicates a clear association between seed geometric characteristics and wheat varieties. 
 Together, PC1 and PC2 account for approximately **88.98%** of the total standardized variance, 
 meaning this two-dimensional plot retains most of the major structure present in the original data.

**Meaning of PC1:** PC1 accounts for approximately **71.9%** of the total variance. 
According to the loading table, area, perimeter, kernel_length, kernel_width, and kernel_groove_length have large absolute loadings on PC1,
and their directions are largely consistent. Therefore, I interpret PC1 as the **"overall seed size axis."** Samples with positive scores on PC1 typically have larger,
longer, or wider seeds, while those with negative scores tend to be smaller in size. 
It should be noted that in PCA, the signs of all principal components can be simultaneously flipped; 
thus, interpretation should focus on the relative directions among variables rather than mechanically equating positive scores with "large."

**The meaning of PC2:** PC2 accounts for approximately **17.1%** of the total variance. 
It is primarily influenced by asymmetry_coefficient and compactness, with these two variables having opposite loadings.
Therefore, I interpret PC2 as the **"axis of seed shape and degree of asymmetry."
** Along one direction of PC2, seeds are more asymmetric and relatively less compact; 
along the other direction, they are more compact and relatively symmetric. 

Thus, while PC1 mainly captures seed size, PC2 further differentiates seed shape, providing complementary information.
"""

SCREE_INTERPRETATION = """
### Interpretation

The bar chart in the scree plot shows the proportion of variance explained by each individual principal component,
 while the line represents the cumulative explained variance as the number of principal components increases. 
 The first two principal components account for most of the major variation in the data; from subsequent components onward, 
 the additional explained variance contributed by each new component decreases significantly. 
 This indicates that two-dimensional PCA is an effective method for capturing the main structure of this dataset, 
 although it does not retain all information. Therefore, when interpreting PCA scatter plots, it is important to also consider the cumulative explained variance.
"""

PAIRWISE_COMPARISON_TEXT = """
**Advantages and limitations of scatterplot matrices:** In a scatterplot matrix, both axes represent original variables, 
preserving their names, units, and actual values, allowing readers to directly interpret the relationship between two measurements. 
It is also better suited for examining nonlinear trends, outliers, and local details within a pair of variables.
 However, a single scatterplot matrix can display only two variables at a time, thereby ignoring information from other numeric variables. 
 When highly correlated variables such as area and perimeter are selected, the two axes may carry substantial redundant information. 
 Therefore, even if a particular scatterplot does not show clear class separation, 
 it does not imply that no grouping structure exists in the complete multivariate dataset.

**Advantages and Limitations of PCA Plots:** 
PCA simultaneously incorporates all numerical variables and identifies two directions that retain the maximum overall variation.
 Thus, compared to examining numerous pairwise combinations individually, 
 PCA more effectively summarizes the distribution and separation of three wheat varieties in terms of their overall geometric characteristics. 
 However, PC1 and PC2 are weighted combinations of multiple original variables, 
 lacking the intuitive units of individual original variables and requiring interpretation in conjunction with a loading plot. 
 Additionally, the two-dimensional PCA retains only about **89.0%** of the total variance, leaving approximately **11.0%** of the information unrepresented.
 Furthermore, PCA emphasizes linear directions with the highest variance, potentially downplaying features with low variance but practical significance, 
 nonlinear structures, or individual outliers.

Overall, paired scatter plots miss the multivariate structure formed by other variables, 
while PCA sacrifices some of the original detail and direct interpretability. 
The two are not mutually exclusive: PCA is suitable for providing an overall overview, 
whereas paired scatter plots are better for returning to specific variables to validate and interpret the structures revealed by PCA.
"""


@st.cache_data
def load_data() -> pd.DataFrame:
    """Download and prepare the UCI Seeds dataset.下载数据库"""
    dataset_url = "https://archive.ics.uci.edu/static/public/236/seeds.zip"
    column_names = [
        "area",
        "perimeter",
        "compactness",
        "kernel_length",
        "kernel_width",
        "asymmetry_coefficient",
        "kernel_groove_length",
        "variety",
    ]#特征名称

    try:
        # Download the ZIP file and open it directly in memory.
        with urlopen(dataset_url, timeout=30) as response:
            zip_bytes = BytesIO(response.read())

        with ZipFile(zip_bytes) as archive:
            with archive.open("seeds_dataset.txt") as data_file:
                data = pd.read_csv(
                    data_file,
                    sep=r"\s+",
                    header=None,
                    names=column_names,
                )
    except Exception as error:
        st.error(f"The dataset could not be downloaded from UCI: {error}")
        st.stop()

    # Replace numeric class codes with meaningful category names.
    variety_names = {1: "Kama", 2: "Rosa", 3: "Canadian"}
    data["variety"] = data["variety"].map(variety_names)
    return data


df = load_data()

# Select all numeric measurements used for correlation and PCA.
numeric_cols = df.select_dtypes(include=np.number).columns.tolist() #数字特征
category_cols = df.select_dtypes(exclude=np.number).columns.tolist()#类别特征

st.title("Week3_hw: Wheat Seed Measurements: Correlation and PCA")
st.caption(
    f"{len(df)} observations · {len(numeric_cols)} numeric variables · "
    "UCI Seeds dataset and this app is a homework of Jinyi Zhou"
)
#control 控制框的属性
with st.sidebar:
    st.header("Chart Controls")

    with st.expander("Heatmap Controls", expanded=True):
        show_labels = st.checkbox(
            "Show correlation values",
            value=True,
            help="Show or hide the correlation coefficient inside each heatmap cell.",
        )

    with st.expander("PCA Controls", expanded=True):
        pca_color_by = st.selectbox(
            "Color PCA points by",
            options=category_cols,
            index=category_cols.index("variety") if "variety" in category_cols else 0,
        )
        pca_point_size = st.slider("PCA point size", 40, 140, 75, 5)
        pca_point_opacity = st.slider(
            "PCA point opacity", 0.30, 1.00, 0.75, 0.05
        )

    with st.expander("Pairwise Controls", expanded=True):
        pairwise_x = st.selectbox("Pairwise x-variable", numeric_cols, index=0)
        default_pairwise_y = 1 if len(numeric_cols) > 1 else 0
        pairwise_y = st.selectbox(
            "Pairwise y-variable", numeric_cols, index=default_pairwise_y
        )
        pairwise_color_by = st.selectbox(
            "Color pairwise points by",
            options=category_cols,
            index=category_cols.index("variety") if "variety" in category_cols else 0,
        )
        pairwise_point_size = st.slider("Pairwise point size", 40, 140, 75, 5)
        pairwise_point_opacity = st.slider(
            "Pairwise point opacity", 0.30, 1.00, 0.75, 0.05
        )

with st.expander("Preview the dataset and assignment variables"):
    st.dataframe(df, use_container_width=True)
    st.write("Numeric variables:", numeric_cols)
    st.write("PCA color category:", pca_color_by)
    st.write("Pairwise color category:", pairwise_color_by)

# -----------------------------------------------------------------------------
# 1. CORRELATION HEATMAP
# -----------------------------------------------------------------------------
st.header("1. Correlation Heatmap")

corr = df[numeric_cols].corr()
corr_long = (
    corr.rename_axis("variable_1")
    .reset_index()
    .melt(id_vars="variable_1", var_name="variable_2", value_name="correlation")
)

heatmap_base = (
    alt.Chart(corr_long)
    .encode(
        x=alt.X("variable_2:N", title=None, sort=numeric_cols),
        y=alt.Y("variable_1:N", title=None, sort=numeric_cols),
        tooltip=[
            alt.Tooltip("variable_1:N", title="Variable 1"),
            alt.Tooltip("variable_2:N", title="Variable 2"),
            alt.Tooltip("correlation:Q", title="Correlation", format=".3f"),
        ],
    )
    .properties(height=500)
)

heatmap_rect = heatmap_base.mark_rect().encode(
    color=alt.Color(
        "correlation:Q",
        title="Pearson r",
        scale=alt.Scale(domain=[-1, 0, 1], range=["#2166ac", "#f7f7f7", "#b2182b"]),
    )
)

if show_labels:
    heatmap_text = heatmap_base.mark_text(fontSize=12).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.correlation) > 0.55",
            alt.value("white"),
            alt.value("black"),
        ),
    )
    heatmap = heatmap_rect + heatmap_text
else:
    heatmap = heatmap_rect
#调整heatmap里面的显示


st.altair_chart(heatmap, use_container_width=True)
st.markdown(HEATMAP_INTERPRETATION)
st.divider()

# -----------------------------------------------------------------------------
# 2. PCA PROJECTION — standardize first, then fit PCA.
# -----------------------------------------------------------------------------
st.header("2. PCA Projection")

# Standardization prevents variables with larger numeric scales from dominating PCA.
scaler = StandardScaler()#标准化
X_scaled = scaler.fit_transform(df[numeric_cols])

pca = PCA()
scores = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    scores[:, :2],
    columns=["PC1", "PC2"],
    index=df.index,
).join(df[category_cols])

pc1_variance = pca.explained_variance_ratio_[0] * 100
pc2_variance = pca.explained_variance_ratio_[1] * 100

variety_color_scale = alt.Scale(
    domain=["Canadian", "Kama", "Rosa"],
    range=["#2E86AB", "#F6AE2D", "#D1495B"],
)

pca_scatter = (
    alt.Chart(pca_df)
    .mark_circle(size=pca_point_size, opacity=pca_point_opacity)
    .encode(
        x=alt.X("PC1:Q", title=f"PC1 ({pc1_variance:.1f}% explained variance)"),
        y=alt.Y("PC2:Q", title=f"PC2 ({pc2_variance:.1f}% explained variance)"),
        color=alt.Color(
            f"{pca_color_by}:N",
            title=pca_color_by.replace("_", " ").title(),
            scale=variety_color_scale,
        ),
        tooltip=[
            alt.Tooltip("PC1:Q", format=".3f"),
            alt.Tooltip("PC2:Q", format=".3f"),
            alt.Tooltip(
                f"{pca_color_by}:N",
                title=pca_color_by.replace("_", " ").title(),
            ),
        ],
    )
    .properties(height=500)
    .interactive()
)

st.altair_chart(pca_scatter, use_container_width=True, theme=None)

# Loadings help explain what PC1 and PC2 represent.
loadings = pd.DataFrame(
    pca.components_[:2].T,
    index=numeric_cols,
    columns=["PC1 loading", "PC2 loading"],
)

with st.expander("PCA loadings — use these to interpret PC1 and PC2", expanded=True):
    st.dataframe(
        loadings.style.background_gradient(
            cmap="RdBu_r", 
            axis=None, 
            vmin=-1, 
            vmax=1)
        .format("{:.3f}")
        .set_properties(
            **{"text-align": "center"}
        ),
        use_container_width=True,
    )
    st.caption(
        "Focus on variables with the largest absolute loadings. The sign shows "
        "direction; PCA signs may be flipped without changing the solution."
    )

st.markdown(PCA_INTERPRETATION)
st.divider()

# -----------------------------------------------------------------------------
# OPTIONAL 1: Scree plot — explained variance for every component.
# -----------------------------------------------------------------------------
st.header("Optional 1: Scree Plot")

scree_df = pd.DataFrame(
    {
        "component_number": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance": pca.explained_variance_ratio_ * 100,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_) * 100,
    }
)

scree_bars = (
    alt.Chart(scree_df)
    .mark_bar(color="#2E6E8E")
    .encode(
        x=alt.X("component_number:O", title="Principal Component"),
        y=alt.Y("explained_variance:Q", title="Explained Variance (%)"),
        tooltip=[
            alt.Tooltip("component_number:O", title="Component"),
            alt.Tooltip("explained_variance:Q", title="Explained variance", format=".2f"),
            alt.Tooltip("cumulative_variance:Q", title="Cumulative variance", format=".2f"),
        ],
    )
)

scree_line = (
    alt.Chart(scree_df)
    .mark_line(point=True, color="#d95f02")
    .encode(
        x=alt.X("component_number:O"),
        y=alt.Y("cumulative_variance:Q", title="Cumulative Variance (%)"),
        tooltip=[alt.Tooltip("cumulative_variance:Q", format=".2f")],
    )
)

scree_chart = alt.layer(scree_bars, scree_line).resolve_scale(y="independent").properties(height=380)#合在一起
st.altair_chart(scree_chart, use_container_width=True)

st.write(
    f"PC1 and PC2 together explain **{pc1_variance + pc2_variance:.1f}%** "
    "of the standardized data's total variance."
)
st.markdown(SCREE_INTERPRETATION)
st.divider()

# -----------------------------------------------------------------------------
# OPTIONAL 2 & 3: Pairwise view and comparison with PCA.
# -----------------------------------------------------------------------------
st.header("Optional 2 & 3: Pairwise View vs. PCA")

pairwise_scatter = (
    alt.Chart(df)
    .mark_circle(size=pairwise_point_size, opacity=pairwise_point_opacity)
    .encode(
        x=alt.X(f"{pairwise_x}:Q", title=pairwise_x.replace("_", " ").title()),
        y=alt.Y(f"{pairwise_y}:Q", title=pairwise_y.replace("_", " ").title()),
        color=alt.Color(
            f"{pairwise_color_by}:N",
            title=pairwise_color_by.replace("_", " ").title(),
            scale=variety_color_scale,
        ),
        tooltip=[
            alt.Tooltip(f"{pairwise_x}:Q", format=".3f"),
            alt.Tooltip(f"{pairwise_y}:Q", format=".3f"),
            alt.Tooltip(f"{pairwise_color_by}:N"),
        ],
    )
    .properties(height=470)
    .interactive()
)

st.altair_chart(pairwise_scatter, use_container_width=True, theme=None)
st.subheader("What does each view miss?")
st.markdown(PAIRWISE_COMPARISON_TEXT)

st.divider()
st.caption(
    "Data source: UCI Machine Learning Repository — Seeds dataset. "
    "The numeric variables were standardized before PCA."
)
