"""Week 4 — Time series: trend, seasonality, uncertainty and animation.
Run: python -m streamlit run week4_app.py
"""
import calendar
import altair as alt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from statsmodels.tsa.seasonal import STL, seasonal_decompose

BLUE, ORANGE, TEAL = '#245B85', '#D97732', '#118579'

@st.cache_data
def load_data():
    # 从 R datasets 下载 AirPassengers 数据
    data = sm.datasets.get_rdataset(
        "AirPassengers",
        package="datasets"
    ).data

    # 设置月度日期索引，根据数据来源来确定
    dates = pd.date_range(
        start="1949-01-01",
        periods=len(data),
        freq="MS"
    )

    # 返回带日期索引的旅客量序列，单位：千人
    return pd.Series(
        data["value"].to_numpy(),
        index=dates,
        name="Passengers",
        dtype=float
    )
@st.cache_data
def bootstrap_paths(block=6, repetitions=1000):#生成很多条可能的旅客量曲线
    """Circular moving-block residual bootstrap, conditional on a fixed log-STL signal.
    Resample normalized multiplicative residuals, NOT trending raw observations.
    Shared sampled paths permit correct aggregation before calculating intervals.
    """
    s = load_data()
    fitted = STL(np.log(s), period=12, seasonal=13, robust=True).fit()#根据数据估计出趋势和季节性
    signal = np.exp(fitted.trend + fitted.seasonal).to_numpy()#基准曲线（模型）
    ratios = s.to_numpy() / signal#真实数据和基准曲线的比例
    signal = signal * ratios.mean()
    ratios = ratios / ratios.mean()  # E*(multiplier) = 1
    rng = np.random.default_rng(6401)
    starts = rng.integers(0, len(s), size=(repetitions, int(np.ceil(len(s)/block))))
    indices = ((starts[..., None] + np.arange(block)) % len(s)).reshape(repetitions, -1)[:, :len(s)]
    return signal, signal[None, :] * ratios[indices]

@st.cache_data
def chart_data(rule, window, block):#按选择的时间粒度整理数据，计算滚动平均和置信区间
    s = load_data()
    signal, paths = bootstrap_paths(block)
    shown = s.resample(rule).mean()#用不同的时间粒度整理数据
    baseline = pd.Series(signal, index=s.index).resample(rule).mean().rolling(window).mean()#基准曲线经过汇总和平滑后的结果
    simulated = pd.DataFrame(paths.T, index=s.index).resample(rule).mean().rolling(window).mean()#模拟的数据
    estimate = shown.rolling(window).mean()#真实的数据
    valid = baseline.notna()
    # Basic bootstrap CI: theta_hat - quantiles(theta_star - theta_reference).
    errors = simulated.loc[valid].to_numpy() - baseline.loc[valid].to_numpy()[:, None]#计算模型偏差（模拟值-基准值）
    low_error, high_error = np.quantile(errors, [0.025, 0.975], axis=1)#95%
    out = pd.DataFrame({'Date': shown.index, 'Observed': shown.values, 'Rolling mean': estimate.values})
    out['Lower'] = np.nan
    out['Upper'] = np.nan
    out.loc[valid.to_numpy(), 'Lower'] = estimate.loc[valid].to_numpy() - high_error#计算下界
    out.loc[valid.to_numpy(), 'Upper'] = estimate.loc[valid].to_numpy() - low_error#计算上届
    return out

def linked_chart(data, zero):
    brush = alt.selection_interval(name='date_brush', encodings=['x'])#添加选择器（用户用鼠标拖出一个范围进行选择）
    base = alt.Chart(data).encode(x=alt.X('Date:T', title='Date'))#创建公共的图表基础
    band = base.mark_area(color=BLUE, opacity=0.20).encode(
        y=alt.Y('Lower:Q', title='Passengers (thousands per month)', scale=alt.Scale(zero=zero)),
        y2='Upper:Q', 
        tooltip=[alt.Tooltip('Date:T', format='%Y-%m'), 
                 alt.Tooltip('Lower:Q', format='.1f'), 
                 alt.Tooltip('Upper:Q', format='.1f')])
    #tooltip（鼠标信息框），mark_area画一个填充区域（置信区间阴影层），y和y2上下边界
    raw = base.mark_line(color=ORANGE, strokeWidth=1.6).encode(y='Observed:Q', tooltip=[alt.Tooltip('Date:T', format='%Y-%m'), alt.Tooltip('Observed:Q', format='.1f')])
    #观测曲线
    trend = base.mark_line(color=BLUE, strokeWidth=3).encode(y='Rolling mean:Q', tooltip=[alt.Tooltip('Date:T', format='%Y-%m'), alt.Tooltip('Rolling mean:Q', format='.1f')])
    #滚动平均线
    detail = alt.layer(band, raw, trend).transform_filter(brush).properties(height=320, title='Detail · observed series, rolling mean and 95% bootstrap CI')
    #把三个图叠成主图
    overview = base.mark_line(color=TEAL).encode(y=alt.Y('Observed:Q', title='Overview', scale=alt.Scale(zero=zero))).add_params(brush).properties(height=85, title='Drag horizontally here to select dates · double-click to reset')
    #总览图把之前的brush添加到总览图上（总览图接受鼠标框选，主图根据框选结果更新显示）
    return alt.vconcat(detail, overview).resolve_scale(x='independent').configure_axis(labelFontSize=11, titleFontSize=12) 
            #vconcat纵向拼接，上图detail，下图overview，resolve_scale(x='independent')横轴独立不影响选择器联动，它只是允许两张图显示不用的日期范围（主图缩放时间轴，总览图保留完整时间轴）


def decomposition(s): #模型结果加绘图
    add = seasonal_decompose(s, model='additive', period=12) #季节分解，加法模型：观测值=趋势+季节影响+残差
    mul = seasonal_decompose(s, model='multiplicative', period=12)#乘法模型：观测值=趋势*季节影响*残差因子
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True)#画布
    for j, (res, name) in enumerate([(add, 'Additive'), (mul, 'Multiplicative')]):#两个模型，第一个先画加法模型，第二次画乘法模型
        for i, (series, label) in enumerate([(s, 'Observed (000s)'), (res.trend, 'Trend (000s)'), (res.seasonal, 'Seasonal effect (000s)' if j == 0 else 'Seasonal factor'), (res.resid, 'Residual (000s)' if j == 0 else 'Residual factor')]):
            #每一次要画的数据和数据纵轴标签（原始观测、趋势、季节项、残差）
            axes[i,j].plot(series.index, series, color=BLUE if j == 0 else TEAL, lw=1.5)#每个区域画图
            axes[i,j].set_ylabel(label)
            axes[i,j].grid(alpha=.15)
        axes[0,j].set_title(name)#每个模型对应的名字
        axes[2,j].axhline(0 if j == 0 else 1, color='gray', ls='--', lw=.8)#水平参考线
        axes[3,j].axhline(0 if j == 0 else 1, color='gray', ls='--', lw=.8)#水平参考线
    fig.tight_layout()
    return fig, add, mul


def draw_animation(s, rolled, window, upto):#画出动画中的一帧（数据、计算好的滚动平均序列、窗口包含多少条月度观测、当前展示到前多少条观测）
    # Adapted from the instructor: same placeholder + draw + close pattern.
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(s.index, s.values, color='#b5bec8', lw=1.2, label='Monthly observations')#作为固定背景
    lo, hi = s.index[upto-window], s.index[upto-1]#起始日期和截至日期
    ax.axvspan(lo, hi, color=BLUE, alpha=.18, label='Calculation window')#阴影标出窗口
    ax.plot(rolled.index[:upto], rolled.values[:upto], color=BLUE, lw=2.3, label=f'{window}-month rolling mean')#滚动平均线
    ax.plot(rolled.index[upto-1], rolled.iloc[upto-1], 'o', color=ORANGE, ms=7)#平均值
    ax.set_xlim(s.index[0], s.index[-1])
    ax.set_ylim(0, s.max()*1.08)
    ax.set_ylabel('Passengers (thousands)')
    ax.set_title(f'{lo:%Y-%m} to {hi:%Y-%m} | mean = {rolled.iloc[upto-1]:.1f} thousand')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    return fig

@st.fragment#当操作这个函数内部的控件时，只重新运行这个区域
def animation_panel(s):#不断改变upto来实现动画播放
    c1, c2 = st.columns(2)#两列控件区域
    window = c1.slider('Animation window (months)', 2, 60, 12, key='anim_window')#滚动窗口长度（最小值、最大值、默认值）
    frames = c2.slider('Animation frames', 10, 80, 40, key='anim_frames')#播放的帧数
    position = st.slider('Inspect an endpoint manually', window, len(s), len(s), key=f'position_{window}')
    play = st.button('▶ Play rolling-window animation')
    rolled = s.rolling(window).mean()#提前算好整条滚动平均线
    slot = st.empty()#预留一个可以替换图片的位置
    if play:
        endpoints = np.unique(np.linspace(window, len(s), min(frames, len(s)-window+1), dtype=int))
        #选出每一帧的窗口终点
        for upto in endpoints:#循环画图并替换上一帧
            fig = draw_animation(s, rolled, window, int(upto))
            slot.pyplot(fig)
            plt.close(fig)
    else:#没有点击播放的时候
        fig = draw_animation(s, rolled, window, position)
        slot.pyplot(fig)
        plt.close(fig)
    st.caption(f'The first {window-1} months have no rolling mean: '
               'the window must be full. Axes remain fixed throughout playback.')
    st.write('Compare windows of 3, 12, and 60: a short window preserves seasonal fluctuations;'
             ' a 12-month window covers one annual cycle; a 60-month window is smoother but also more lagged. '
             'The vertical shaded areas here represent the calculation window, not confidence intervals.')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title='Week 4 | Airline passengers', layout='wide')
    st.title('Airline passengers: growth, seasons & uncertainty')
    st.caption('DATS 6401 · Week 4 | 1949–1960 | Monthly international airline passenger totals')
    s = load_data()
    a,b,c = st.columns(3)
    a.metric('Coverage', '12 years')
    b.metric('Monthly observations', len(s))
    c.metric('Missing months', int(s.isna().sum()))
    st.markdown('**Research question:** How did passenger traffic grow, how did the annual seasonal pattern change, and how sensitive is the story to time resolution? Values are in thousands; data are embedded for reproducibility.')
    with st.expander('Data and provenance'):
        st.write('AirPassengers is the classic Box–Jenkins monthly international '
                 'airline passenger series. '
                 'Source cited by R: Box, Jenkins & Reinsel (1994), '
                 'Time Series Analysis: Forecasting and Control, '
                 '3rd edition. No interpolation or missing-value replacement is needed.')
        st.dataframe(s.rename_axis('Date').reset_index(), hide_index=True)
        st.download_button('Download monthly CSV', s.to_csv().encode(), 'airpassengers_monthly.csv', 'text/csv')
    st.divider()
    st.subheader('1 · Trend, seasonality and bootstrap uncertainty')
    c1,c2,c3 = st.columns(3)
    label = c1.selectbox('Resolution — main chart only', ['Monthly', 'Quarterly', 'Yearly'])
    rule, unit, default, maximum = {'Monthly':('MS','months',12,36),'Quarterly':('QS','quarters',4,12),'Yearly':('YS','years',2,6)}[label]#创建字典（当选择不同resolution的时候，采用不同的情况）
    window = c2.slider(f'Rolling window ({unit})', 1, maximum, default, key=f'window_{rule}')
    block = c3.selectbox('Bootstrap block length (months)', [3,6,12], index=1)
    zero = st.checkbox('Start main-chart y-axis at zero', value=True)
    data = chart_data(rule, window, block)
    st.markdown('**Orange:** observations · '
                '**Blue:** trailing rolling mean · '
                '**Blue band:** approximate pointwise 95% bootstrap confidence interval for the expected rolling average.')
    st.info('In the green overview chart below, hold down the mouse and drag horizontally to select a date range;'
            ' the main chart above will zoom to the corresponding area. '
            'Double-clicking the overview chart resets it. '
            'Selecting a range only changes the view of the main chart—it does not refit the model or alter the subsequent full monthly analysis.')
    st.altair_chart(linked_chart(data, zero), width='stretch')
    st.caption(f'{label} resolution; arithmetic mean of monthly totals per bin (not quarterly/yearly totals).'
               ' Trailing window = {window} {unit}. Bin dates label period starts.'
               ' Bootstrap: 1,000 replicates; {block}-month circular residual blocks;'
               ' seed 6401. Y-axis '+
               ('starts at zero.' if zero else 'is truncated; read numerical ticks carefully.'))
    st.write('The monthly curve retains both long-term growth and intra-year fluctuations. '
             'Quarterly aggregation dampens seasonal variations, while annual aggregation masks the seasonality within the year. '
             'Rolling averages use current and prior observations, thus introducing lag;'
             ' averages or intervals are not plotted when there is insufficient data for a complete window in earlier periods.')
    with st.expander('Bootstrap method and limitations — Self-service method interval explanation'):
        st.markdown('''1. Fit STL to **log monthly traffic**, with annual period 12; keep its estimated trend and seasonal signal fixed.
2. Divide observations by the fitted positive signal and normalize these residual multipliers to mean 1.
3. Resample **consecutive circular blocks** of multipliers (1,000 replicates), preserving some short-range dependence. Multiply by the fixed signal to reconstruct monthly pseudo-series.
4. Apply the **same resolution and rolling window** to each pseudo-series. Let `error* = rolling(pseudo) − rolling(fitted signal)`.
5. The basic 95% interval is `[observed rolling mean − q97.5(error*), observed rolling mean − q2.5(error*)]`.

This is an **approximate, conditional, pointwise confidence interval for the expected trailing average** under the fitted residual model. It is not ±2σ, a prediction interval, a simultaneous confidence band, or a forecast. It assumes normalized residuals are approximately stationary and that the chosen blocks capture relevant dependence. It does **not** include uncertainty from estimating the STL signal, structural breaks, or the choice of smoother. Try different block lengths as a sensitivity check.
                    ''')
                    
                    
    st.divider()
    st.subheader('2 · Annual seasonality — full monthly data')
    df = s.rename_axis('Date').reset_index()#给索引命名为date，并变成普通列
    df['Year'] = df.Date.dt.year.astype(str)#用str作为类别
    df['Month'] = df.Date.dt.month#添加新的一列
    df['Relative to annual mean'] = df.Passengers / df.groupby('Year').Passengers.transform('mean')
   #计算相对于本年度均值的旅客量
    kind = st.radio('Seasonal view', ['Passenger counts', 'Relative to annual mean'], horizontal=True)#创建两个单选项
    field = 'Passengers' if kind == 'Passenger counts' else 'Relative to annual mean'#纵坐标标签
    seasonal = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Month:O', sort=list(range(1,13)), title='Calendar month (1 = January)'),
        y=alt.Y(f'{field}:Q', scale=alt.Scale(zero=True)), color=alt.Color('Year:N', scale=alt.Scale(scheme='viridis')),
        tooltip=['Year:N','Month:O',alt.Tooltip(f'{field}:Q',format='.2f')]).properties(height=300)
    st.altair_chart(seasonal, width='stretch')
    peak = df.groupby('Month')['Relative to annual mean'].mean().idxmax()
    trough = df.groupby('Month')['Relative to annual mean'].mean().idxmin()
    st.write('Across years, the average normalized seasonal profile peaks in '
             f'**{calendar.month_name[peak]}** and is lowest in'
             f' **{calendar.month_name[trough]}**. '
             'Absolute seasonal swings grow with traffic, '
             'suggesting a multiplicative structure. '
             'Annual normalization helps compare shapes,'
             ' although within-year trend can still affect them.')
    
    st.write('Each line represents a year. '
             'Switching to relative annual averages reduces the impact of overall level differences between years,'
             ' allowing for comparison of the shapes of peak and off-peak seasons.')
    st.divider()
    
    
    st.subheader('3 · Additive versus multiplicative decomposition')
    st.caption('Both models use all 144 monthly observations and period = 12, independent of the main-chart resolution and brush.')
    fig, add, mul = decomposition(s)
    st.pyplot(fig)
    plt.close(fig)
    fits = pd.DataFrame({'Observed':s, 'Additive':add.trend+add.seasonal, 'Multiplicative':mul.trend*mul.seasonal}).dropna()
    metrics=[]
    for name in ['Additive','Multiplicative']:
        errors=fits.Observed-fits[name]
        metrics.append({'Model':name,'RMSE (thousand passengers)':np.sqrt(np.mean(errors**2)), 'MAE (thousand passengers)':np.mean(np.abs(errors)), 'Compared months':len(fits)})
    table=pd.DataFrame(metrics)
    st.dataframe(table.round(2), hide_index=True)
    winner=table.loc[table['RMSE (thousand passengers)'].idxmin(),'Model']
    st.write(f'**Interpretation:** {winner} has the lower in-sample reconstruction RMSE on the same interior months. '
             'These are descriptive fit diagnostics, not out-of-sample prediction scores. '
             'Additive assumes a fixed seasonal amount; multiplicative assumes a seasonal proportion of the changing level.')
    st.write('Additive: Observation = Trend + Seasonal Component + Residual; '
             'Multiplicative: Observation = Trend × Seasonal Factor × Residual Factor. '
             'The neutral value for multiplicative seasonality/residuals is 1, while for additive it is 0.'
             ' Since the two original residual units differ, residuals cannot be directly compared in magnitude; '
             'the table above converts fitted values back to the same passenger volume unit before comparison.')
    
    st.caption('Classical centered trend estimation leaves six missing months at each edge. These are intentionally retained as gaps.'
               ' Charts use separate labeled component scales; inspect residual structure as well as the common-scale errors.')
    
    
    st.divider()
    st.subheader('4 · Watch the rolling window move')
    st.caption('Adapted from the instructor’s animation. These controls apply only to this monthly animation; the upper chart’s brush and resolution do not affect it.')
    animation_panel(s)
    
    
    
    st.divider()
    st.subheader('5 · Temporal honesty — design note')
    st.markdown(f'''I preserve a regular monthly calendar and label dates at the start of each aggregation period. '
                'The main chart currently uses **{label.lower()} means of monthly passenger totals**, keeping units comparable across resolutions.'
                ' Annual averaging hides within-year seasonality, so the seasonal and decomposition panels always retain monthly data. '
                'I disclose the trailing window (**{window} {unit}**), its lag and its missing initial values. '
                'The main y-axis **{'starts at zero' if zero else 'is explicitly truncated'}**; animation axes remain fixed.'
                ' Brushing changes the displayed interval without silently refitting the model. '
                'The shaded horizontal band is explicitly a conditional bootstrap confidence interval; '
                'the animation’s vertical band is only a calculation window.''')
    
    st.write('My time chart honesty principles: clearly label time granularity, aggregation method, and window; '
             'avoid mislabeling annual averages as annual totals; refrain from using animated auto-scaling to exaggerate changes; '
             'do not confuse confidence intervals with the range of raw data fluctuations.')
    
    st.markdown('**Conclusion:** Passenger traffic increases over the 12 years and repeats a strong annual cycle. '
                'Increasing seasonal amplitude favors a proportional (multiplicative) description. '
                'Resolution and smoothing change which features remain visible; '
                'the bootstrap band adds conditional uncertainty without claiming to cover future observations.')
    
    with st.expander('Sources and implementation credits'):
        st.markdown(''' — original series description and bibliography.
- [CSV mirror used for the embedded values](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv).
- [Statsmodels seasonal decomposition](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html).
- [Altair interval selection](https://altair-viz.github.io/user_guide/interactions/parameters.html).
- Instructor starter files: `app_resolution_starter.py` and `app_rolling_animation.py`. The animation retains the fragment / placeholder / redraw pattern; manual inspection and exact endpoint sampling were added.
- Bootstrap implementation: circular residual-block resampling with a fixed log-STL pilot signal; assumptions and limitations are documented above.''')

if __name__ == '__main__':
    main()
