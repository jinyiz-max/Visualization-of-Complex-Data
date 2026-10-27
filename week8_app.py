"""Week 8 image exploration app using the public EuroSAT RGB dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image, ImageFilter #图像处理工具


st.set_page_config(page_title="🛰EuroSAT Image Browser", layout="wide")
st.title("EuroSAT Image Browser")
st.caption("Week 8: satellite image montage, RGB channel/intensity structure, and edge feature view by Jinyi Zhou")
#设置浏览器标签上的标题以及title和caption

DATA_DIR = Path("data/eurosat")
ZIP_PATH = DATA_DIR / "EuroSAT_RGB.zip"
EXTRACT_DIR = DATA_DIR / "EuroSAT_RGB"
DATA_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
DISPLAY = {
    "AnnualCrop": "Annual crop", "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous vegetation", "Highway": "Highway",
    "Industrial": "Industrial area", "Pasture": "Pasture",
    "PermanentCrop": "Permanent crop", "Residential": "Residential area",
    "River": "River", "SeaLake": "Sea or lake",
}
#对应的数据下载路径，以及在网页中展示的内容（class——十种土地覆盖类别）

@st.cache_resource(show_spinner=False)
def ensure_dataset() -> str:
    #下载并解压 EuroSAT RGB 压缩包
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    #创建存放数据的文件夹
    if not any(EXTRACT_DIR.rglob("*.jpg")):
        if not ZIP_PATH.exists():
            with urlopen(DATA_URL, timeout=180) as response:
                ZIP_PATH.write_bytes(response.read())
                #数据包不在就下载
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATA_DIR)
            #打开压缩包解压内容
    roots = [p for p in DATA_DIR.iterdir() if p.is_dir() and any(p.rglob("*.jpg"))]
    if not roots:
        raise FileNotFoundError("The archive was extracted, but no JPG images were found.")
    return str(roots[0])
    #返回符合条件的文件夹路径


@st.cache_data(show_spinner=False)
def list_images(root: str) -> dict[str, list[str]]:
    #按类别整理图片路径
    base = Path(root)
    result: dict[str, list[str]] = {}
    for cls in CLASSES:
        matches = list((base / cls).glob("*.jpg"))
        #依次处理十个类别
        if not matches:
            matches = list(base.rglob(f"{cls}/*.jpg"))
        result[cls] = sorted(str(p) for p in matches)
        #将图片路径转换为字符串，排序后存入字典
    return result
    


@st.cache_data(show_spinner=False)
def read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
    #将图片转换为RGB三通道格式，然后再转换为Numpy数组


def make_montage(paths: list[str], n: int, title: str) -> plt.Figure:
    #将多张图片拼成网格
    selected = paths[:n]
    cols = min(6, n)
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.1), squeeze=False)
    #创建网格画布
    for ax, path in zip(axes.flat, selected):
        ax.imshow(read_rgb(path))
        ax.set_title(Path(path).stem, fontsize=7)
        ax.axis("off")
        #遍历图片
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig
    #同时观察同一类别内部的颜色、空间布局等差异


@st.cache_data(show_spinner=False)
def edge_map(path: str) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGB")
    edge = image.filter(ImageFilter.FIND_EDGES).convert("L")
    #原图和边缘图
    return np.asarray(image), np.asarray(edge, dtype=np.float32) / 255.0


with st.expander("About this dataset", expanded=False):
    st.markdown(
        """
**EuroSAT RGB** contains 27,000 labeled satellite image patches based on Sentinel-2 observations. The RGB release has 10 land-use/land-cover classes; each image is a 64×64 JPEG using red, green, and blue bands.

**Official sources:** [EuroSAT research repository](https://github.com/phelber/EuroSAT) · [Zenodo dataset record and download](https://zenodo.org/records/7711810)

Images are small patches labeled with a broad land-cover category. RGB pixel values are display values, not the full 13-band scientific measurements in the multispectral release. Classes can have visually similar patterns, and a single patch does not describe an entire region.

**Classes:** Annual crop, forest, herbaceous vegetation, highway, industrial area, pasture, permanent crop, residential area, river, sea or lake.
        """
    )
    #介绍数据来源、图片尺寸

with st.sidebar:
    st.header("Image browser")
    selected_class = st.selectbox("Class", CLASSES, format_func=lambda c: DISPLAY[c])
    montage_n = st.slider("Images in montage", 6, 30, 12, step=6)
    #用户选择类别和图片数量

try:
    with st.spinner("Downloading and preparing EuroSAT RGB on first run (~95 MB)…"):
        root = ensure_dataset()
        paths_by_class = list_images(root)
        #加载数据
except Exception as exc:
    st.error(f"Could not load EuroSAT: {exc}")
    st.stop()

counts = {c: len(paths_by_class[c]) for c in CLASSES}
total = sum(counts.values())
st.caption(f"Loaded **{total:,}** images across **{sum(v > 0 for v in counts.values())}** classes · selected class: **{DISPLAY[selected_class]}** ({counts[selected_class]:,} images)")
#统计图片数量
if total == 0:
    st.error("No images were found. Check the dataset download and extraction.")
    st.stop()

tab_montage, tab_structure, tab_feature, tab_compare = st.tabs(
    ["Montage", "Channels & intensity", "Feature view", "Class comparison"]
)

with tab_montage:#展示同类图片
    st.subheader(f"{DISPLAY[selected_class]} montage")
    st.write("The montage reveals within-class variation in color, texture, and spatial layout. The category stays the same even when the visible scene patterns differ.")
    fig = make_montage(paths_by_class[selected_class], montage_n, DISPLAY[selected_class])
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with tab_structure:#分析RGB通道和亮度
    st.subheader("Structural visualization: RGB channels and intensity")
    st.write("Unlike grayscale datasets, EuroSAT RGB has three channels. Comparing them shows which colors dominate a scene; the intensity map summarizes brightness across pixels.")
    path = st.selectbox("Choose an image", paths_by_class[selected_class], key="structure_image", format_func=lambda p: Path(p).stem)
    rgb = read_rgb(path)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]#拆分RGB
    channel_fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(rgb); axes[0].set_title("RGB composite")
    #四个子图：原始、红绿蓝通道
    for ax, channel, name, cmap in zip(axes[1:], [r, g, b], ["Red", "Green", "Blue"], ["Reds", "Greens", "Blues"]):
        ax.imshow(channel, cmap=cmap, vmin=0, vmax=255); ax.set_title(f"{name} channel")
    
    for ax in axes: ax.axis("off")
    channel_fig.tight_layout()
    st.pyplot(channel_fig)
    plt.close(channel_fig)
    
    intensity = rgb.mean(axis=2)
    hist_fig, ax = plt.subplots(figsize=(7, 3))
    for channel, name, color in [(r, "Red", "red"), (g, "Green", "green"), (b, "Blue", "blue")]:
        sns.histplot(channel.ravel(), bins=32, binrange=(0, 255), stat="density", element="step", fill=False, ax=ax, color=color, label=name)
    #绘制三个通道的像素值分布
    ax.set(xlabel="Channel value (0–255)", ylabel="Pixel density", title="RGB channel intensity distributions")
    ax.legend()
    st.pyplot(hist_fig)
    plt.close(hist_fig)
    st.caption(f"Mean grayscale intensity for this patch: {intensity.mean():.1f} on a 0–255 scale. RGB channels are correlated because they come from the same scene.")

with tab_feature:#展示边缘特征
    st.subheader("Feature view: edge strength")
    st.caption("This is a simple image-processing feature map, not a trained model's saliency map. It highlights local brightness/color boundaries such as field edges, roads, shorelines, and built structures.")
    path = st.selectbox("Choose a class example", paths_by_class[selected_class], key="feature_image", format_func=lambda p: Path(p).stem)
    orig, edges = edge_map(path)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(orig); axes[0].set_title("Original")
    axes[1].imshow(edges, cmap="magma", vmin=0, vmax=1); axes[1].set_title("Edge strength")
    axes[2].imshow(orig); axes[2].imshow(edges, cmap="magma", alpha=0.55, vmin=0, vmax=1); axes[2].set_title("Overlay")
    #帮助理解图片中的纹理与边界信息
    for ax in axes: 
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write("Interpretation: stronger edges mark sharper transitions, which can reveal texture and boundaries. This does not show what a classifier would use to make a prediction.")

with tab_compare:
    st.subheader("Optional extensions: class balance and intensity comparison")
    balance = pd.DataFrame({"Class": [DISPLAY[c] for c in CLASSES], "Images": [counts[c] for c in CLASSES]})
    st.bar_chart(balance.set_index("Class"))
    #将类别和图片数量整理成表格，再绘制柱状图。
    col1, col2 = st.columns(2)
    with col1:
        class_a = st.selectbox("First class", CLASSES, index=0, key="class_a", format_func=lambda c: DISPLAY[c])
    with col2:
        class_b = st.selectbox("Second class", CLASSES, index=1, key="class_b", format_func=lambda c: DISPLAY[c])
    # Sample a maximum of 300 images per class to keep rendering quick.
    values, group = [], []
    for cls in [class_a, class_b]:
        selected = paths_by_class[cls][:300]
        means = [read_rgb(p).mean(axis=2).mean() / 255.0 for p in selected]
        values.extend(means)
        group.extend([DISPLAY[cls]] * len(means))
    #计算每张图片平均亮度
    dist = pd.DataFrame({"Mean image intensity (0–1)": values, "Class": group})
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(data=dist, x="Mean image intensity (0–1)", hue="Class", bins=25, stat="density", common_norm=False, element="step", fill=False, ax=ax)
    ax.set_title("Distribution of image-level mean intensity")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Each observation is one image's average RGB brightness. Differences may reflect land cover, lighting, season, or image composition; they do not by themselves establish a causal explanation.")
