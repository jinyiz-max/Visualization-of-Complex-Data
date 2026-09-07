# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:08:13 2026

@author: ZJY
"""

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Palmer Penguins: Comparing Visual Encodings",
    page_icon="🐧",
    layout="wide",
)
#app的总体标题

DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/"
    "palmerpenguins/main/inst/extdata/penguins.csv"
)

SPECIES_COLORS = {
    "Adelie": "#0072B2",
    "Chinstrap": "#D55E00",
    "Gentoo": "#009E73",
}


@st.cache_data #表示让 Streamlit 缓存读取好的数据。Streamlit 页面每次互动时都会重新运行代码，如果没有缓存，它可能反复读取同一个 CSV
def load_data() -> pd.DataFrame: 
    """Load and clean the variables used in the visualizations."""
    data = pd.read_csv(DATA_URL)
    columns = ["species", "flipper_length_mm", "body_mass_g"]
    return data.dropna(subset=columns).copy()


def base_chart(data: pd.DataFrame) -> alt.Chart: #固定横轴、纵轴的范围 tooltip是当鼠标移到对应的点会展示对应的数据
    """Create a shared chart base so all designs use comparable axes."""
    return (
        alt.Chart(data)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X(
                "flipper_length_mm:Q",
                title="Flipper Length (mm)",
                scale=alt.Scale(domain=[170, 235]),
            ), #Q表示数值型变量 N表示无顺序的分类变量 O表示有顺序 T表示时间变量
            y=alt.Y(
                "body_mass_g:Q",
                title="Body Mass (g)",
                scale=alt.Scale(domain=[2500, 6500]),
            ),
            tooltip=[
                alt.Tooltip("species:N", title="Species"),
                alt.Tooltip("flipper_length_mm:Q", title="Flipper length (mm)"),
                alt.Tooltip("body_mass_g:Q", title="Body mass (g)"),
            ],
        )
        .properties(height=430)#设置图表高度
        .interactive()#允许用户缩放和移动图表
    )


try:
    penguins = load_data()
except Exception as error:
    st.error(
        "The dataset could not be loaded. Download `penguins.csv` from the "
        "Palmer Penguins repository and place it in the same folder as `app.py`."
    )
    st.exception(error)
    st.stop()


st.title("🐧 Palmer Penguins: Comparing Visual Encodings")#主标题
st.caption("DATS 6401 — Week 2: Grammar of Graphics & Design Principles by Jinyi zhou")#说明文字

st.markdown(
    """
### Research question

**How does the relationship between flipper length and body mass differ across
penguin species?**

The three charts below address the same question with the same quantitative
encodings: flipper length is mapped to horizontal position and body mass is
mapped to vertical position. Only the visual encoding of **species** changes.
This controlled comparison makes it possible to evaluate color, shape, and
faceting using perceptual-accuracy and Gestalt principles.
"""
)

metric_1, metric_2, metric_3 = st.columns(3) #分成三列
metric_1.metric("Observations", f"{len(penguins):,}")
metric_2.metric("Species", penguins["species"].nunique())
metric_3.metric("Missing rows removed", 344 - len(penguins))

with st.expander("Dataset and grammar-of-graphics specification"):#创建可展开和收起的区域
    st.markdown(
        """
The Palmer Penguins dataset contains measurements collected for three penguin
species in the Palmer Archipelago, Antarctica. Rows missing any variable used
in this analysis are removed.

| Grammar component | Specification |
|---|---|
| Data | Individual penguin observations |
| Mark | Point |
| X encoding | Flipper length (quantitative) → horizontal position |
| Y encoding | Body mass (quantitative) → vertical position |
| Third variable | Species (nominal) → color, shape, or facet |

Source: [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)
"""
    )
    st.dataframe(penguins.head(10), use_container_width=True, hide_index=True)
#用with进行划分区域

st.divider()#添加分隔线

st.header("Design 1 — Species encoded by color")#图标题

color_chart = (
    base_chart(penguins)
    .encode(
        color=alt.Color(
            "species:N",
            title="Species",
            scale=alt.Scale(
                domain=list(SPECIES_COLORS.keys()),
                range=list(SPECIES_COLORS.values()),
            ),
        )
    )
    .properties(title="Body Mass vs. Flipper Length — Color Encoding")
)
st.altair_chart(color_chart, use_container_width=True)

st.markdown(
    """
**Encoding justification.** Position on a common scale is used for the two
quantitative variables because people judge position more accurately than
area, angle, or color intensity. Species is nominal rather than ordered, so
distinct hues are appropriate: the colors separate categories without implying
that one species is “greater” than another.

**Perception and design principle.** Color hue is a **preattentive feature**,
so the three species can be detected quickly before deliberate inspection.
Similarity, a **Gestalt principle**, also causes points with the same color to
be perceived as a group. A colorblind-friendly blue–orange–green palette and
directly labeled axes improve accessibility. One limitation is that this
design depends on color perception and requires the reader to consult a legend.
"""
)

st.divider()

st.header("Design 2 — Species encoded by shape")

shape_chart = (
    base_chart(penguins)
    .mark_point(filled=True, size=90, opacity=0.72, color="#334155")
    .encode(
        shape=alt.Shape(
            "species:N",
            title="Species",
            scale=alt.Scale(
                domain=["Adelie", "Chinstrap", "Gentoo"],
                range=["circle", "square", "triangle-up"],
            ),
        )
    )
    .properties(title="Body Mass vs. Flipper Length — Shape Encoding")
)
st.altair_chart(shape_chart, use_container_width=True)

st.markdown(
    """
**Encoding justification.** The horizontal and vertical positions remain
unchanged, while species is represented by three distinct shapes. Shape is
valid for nominal categories because it distinguishes groups without suggesting
an order. It also remains interpretable in grayscale, making it useful when
color is unavailable.

**Perception and design principle.** The chart uses the Gestalt principle of
**similarity**: points with the same shape are grouped mentally. However, small
shapes are less preattentive and usually slower to distinguish than saturated
color hues. Overlapping points also make shape boundaries harder to recognize.
Therefore, this encoding is accessible for grayscale reproduction but less
efficient for rapid pattern detection than Design 1.
"""
)

st.divider()

st.header("Design 3 — Species encoded by faceting")

facet_chart = (
    alt.Chart(penguins)
    .mark_circle(size=65, opacity=0.72, color="#0072B2")
    .encode(
        x=alt.X(
            "flipper_length_mm:Q",
            title="Flipper Length (mm)",
            scale=alt.Scale(domain=[170, 235]),
        ),
        y=alt.Y(
            "body_mass_g:Q",
            title="Body Mass (g)",
            scale=alt.Scale(domain=[2500, 6500]),
        ),
        column=alt.Column(
            "species:N",
            title=None,
            header=alt.Header(labelFontSize=15, labelFontWeight="bold"),
            sort=["Adelie", "Chinstrap", "Gentoo"],
        ),
        tooltip=[
            alt.Tooltip("species:N", title="Species"),
            alt.Tooltip("flipper_length_mm:Q", title="Flipper length (mm)"),
            alt.Tooltip("body_mass_g:Q", title="Body mass (g)"),
        ],
    )
    .properties(width=250, height=360)
    .resolve_scale(x="shared", y="shared")
)
st.altair_chart(facet_chart, use_container_width=True)

st.markdown(
    """
**Encoding justification.** Faceting assigns each species to a separate panel
while preserving identical x- and y-axis scales. This removes overlap between
species and makes each within-species relationship easy to inspect. Shared
scales are essential; independent scales would make cross-species comparisons
misleading.

**Perception and design principle.** Faceting uses the Gestalt principles of
**common region** and **proximity**. Points inside the same panel are immediately
perceived as belonging together. The separation reduces clutter, but it also
forces the reader to move attention between panels and remember positions.
Consequently, within-species trends are clearer, while direct comparison across
species is somewhat slower than in the single-panel color chart.
"""
)

st.divider()

st.header("Design 4 — Species encoded by color and shape")

color_shape_chart = (
    base_chart(penguins)
    .mark_point(
        filled=True,
        size=90,
        opacity=0.72,
    )
    .encode(
        shape=alt.Shape(
            "species:N",
            title="Species",
            scale=alt.Scale(
                domain=["Adelie", "Chinstrap", "Gentoo"],
                range=["circle", "square", "triangle-up"],
            ),
        ),
        color=alt.Color(
            "species:N",
            title="Species",
            scale=alt.Scale(
                domain=list(SPECIES_COLORS.keys()),
                range=list(SPECIES_COLORS.values()),
            ),
        ),
    )
    .properties(title="Body Mass vs. Flipper Length — color and shape Encoding")
)
st.altair_chart(color_shape_chart, use_container_width=True)

st.markdown(
    """
**Encoding justification.** This design uses redundant encoding by mapping
species to both color and shape. The two channels communicate the same
categorical variable rather than representing two different variables.

**Perception and design principle.** Color provides fast preattentive grouping,
while shape provides a second way to identify species. This improves
accessibility because readers who have difficulty distinguishing colors can
still use the shapes. The design also also applies the Gestalt principle of
similarity: points with the same color and shape are perceived as belonging to
the same group. However, using two channels for the same variable adds visual
complexity and may be unnecessary when the color encoding alone is already
clear.
"""
)

st.divider()

st.header("Comparison and Conclusion")

st.markdown(
    """
| Design | Main advantage | Main limitation | Best use |
|---|---|---|---|
| Color | Provides fast, preattentive grouping in one shared plot | Relies on color perception and requires a legend | Rapid overall comparison |
| Shape | Works without color and remains readable in grayscale | Shapes are slower to identify and may be obscured by overlapping points | Grayscale reproduction |
| Faceting | Separates species and removes between-group overlap | Requires the reader to compare across separate panels | Examining within-species patterns |
| Color and shape | Provides two visual cues and improves accessibility | Adds some visual complexity through redundant encoding | Clear and accessible overall comparison |

### Conclusion

All four designs use horizontal and vertical position for the two quantitative
variables because position on a common scale supports accurate comparison.
The difference among the designs is how the nominal variable, species, is
represented.

Among the four designs, **Design 4 is the most effective overall**. It uses
redundant encoding by mapping species to both color and shape. Color supports
fast preattentive grouping, while shape provides an additional visual cue.
Readers can therefore distinguish the three species even if they have difficulty
perceiving some colors or if the chart is reproduced in grayscale.

Design 4 also applies the Gestalt principle of similarity: points with the same
color and shape are naturally perceived as belonging to the same group.
Although redundant encoding adds slightly more visual complexity, the dataset
contains only three species, so the chart remains readable rather than cluttered.

The color-only design is also effective and visually simple, but it depends more
heavily on color perception. The shape-only design is accessible without color,
but it takes longer to interpret. Faceting clearly separates the species, but
comparison is slower because the reader must look across different panels.

Overall, Design 4 provides the best balance of **clarity, perceptual speed, and
accessibility**. Across all four designs, the data show a positive relationship
between flipper length and body mass. Gentoo penguins generally have longer
flippers and greater body mass, while Adelie penguins generally appear in the
lower flipper-length and body-mass region.
"""
)
st.info(
    "Use the chart controls to zoom or pan, and hover over any point to see "
    "its species, flipper length, and body mass.",
    icon="💡"
)

