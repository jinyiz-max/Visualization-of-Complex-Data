from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

ENDPOINT = "https://api.open-meteo.com/v1/forecast"
#API 地址
CITIES = {
    "Washington, DC": (38.9072, -77.0369, "America/New_York"),
    "Chicago, IL": (41.8781, -87.6298, "America/Chicago"),
    "Seattle, WA": (47.6062, -122.3321, "America/Los_Angeles"),
}
#字典，每个城市对应不同的纬度、经度、时区
FIELDS = ("temperature_2m", "precipitation_probability")
#想从API中获取的天气数据，两米处的气温、降水概率

def forecast_parameters(latitude, longitude, tz, days):
    return {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": tz,
        "forecast_days": days,
        "hourly": ",".join(FIELDS),
        "temperature_unit": "fahrenheit",
    }
#接受位置时区和预报天数，返回一个包含API参数的字典

@st.cache_data(ttl=600, max_entries=12, show_spinner=False)#缓存天气数据
def fetch_forecast(latitude, longitude, tz, days):
    response = requests.get(
        ENDPOINT,
        params=forecast_parameters(latitude, longitude, tz, days),
        timeout=(4, 8),
    )
    #获取数据，根据API输入对应的参数获取天气数据
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";")[0]
    if content_type.strip().lower() != "application/json":
        raise ValueError("Expected a JSON response.")
    #确认API返回JSON
    payload = response.json()
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    #读取hourly的内容
    if not isinstance(hourly, dict):
        raise ValueError("Missing hourly forecast data.")
        
        
    columns = [hourly.get(name) for name in ("time", *FIELDS)]
    if any(not isinstance(column, list) for column in columns):
        raise ValueError("Missing hourly forecast columns.")
    if not columns[0] or len({len(column) for column in columns}) != 1:
        raise ValueError("Hourly forecast columns are empty or misaligned.")
    frame = pd.DataFrame(dict(zip(("time", *FIELDS), columns)))
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    #将数据整理成dataframe（time、temperature_2m、precipitation_porbability）,time转成datetime类型
    for field in FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    #转成数值类型
    if frame["time"].isna().any() or frame[list(FIELDS)].isna().all().any():
        raise ValueError("The forecast does not contain usable values.")
    return frame.sort_values("time"), datetime.now(timezone.utc)
    #按时间排序的天气表格、获取数据时的 UTC 时间

def forecast_chart(frame, field, label, local_time, color, bars=False):
    #frame	天气数据表格、field	要画哪一列、label	纵轴标题
    #local_time	显示在横轴标题中的时区、color	图表颜色、bars	是否画柱状图，默认画折线图
    #气温图画折线图、降水量画bar图
    chart = alt.Chart(frame)
    if bars:
        chart = chart.mark_bar(color=color, opacity=0.78)
    else:
        chart = chart.mark_line(color=color, strokeWidth=3)
    scale = alt.Scale(domain=[0, 100]) if bars else alt.Scale(zero=False)
    return chart.encode(
        x=alt.X("time:T", title=f"Local time ({local_time})", axis=alt.Axis(format="%b %d %H:%M")),
        y=alt.Y(f"{field}:Q", title=label, scale=scale),
        tooltip=[
            alt.Tooltip("time:T", title="Local time", format="%b %d, %H:%M"),
            alt.Tooltip(f"{field}:Q", title=label, format=".1f"),
        ],
    ).properties(height=195)
    #画图，tooltip悬停展示

def main():
    st.set_page_config(page_title="Hourly weather forecast", layout="wide")
    st.title("Hourly weather forecast")
    city_column, days_column = st.columns(2)
    city = city_column.selectbox("City", CITIES)
    days = days_column.selectbox("Forecast days", (1, 2, 3), index=1)
    #selectbox给予选择
    latitude, longitude, tz = CITIES[city]
    parameters = forecast_parameters(latitude, longitude, tz, days)
    st.subheader("API endpoint and request parameters")
    st.code(f"GET {ENDPOINT}", language="text")
    st.dataframe(
        pd.DataFrame({"Parameter": parameters.keys(),
                      "Value": [str(value) for value in parameters.values()]}),
        hide_index=True,
        use_container_width=True,
    )
    #展示API和对应的参数
    with st.expander("Full request URL"):
        st.code(requests.Request("GET", ENDPOINT, params=parameters).prepare().url,
                language="text", wrap_lines=True)

    try:
        with st.spinner("Fetching hourly forecast…"):
            frame, fetched_at = fetch_forecast(latitude, longitude, tz, days)
    except requests.Timeout:
        st.error("Open-Meteo took too long to respond. Try again in a moment.")
    except requests.HTTPError as error:
        st.error(f"Open-Meteo returned HTTP {error.response.status_code}. Try again in a moment.")
    except requests.ConnectionError:
        st.error("Could not connect to Open-Meteo. Check your internet connection and retry.")
    except (requests.RequestException, ValueError):
        st.error("Open-Meteo did not return a usable forecast. Try again in a moment.")
    #处理失败情况
    else:
        st.caption(
            f"{city} · {len(frame)} hourly forecast values · "
            f"Fetched {fetched_at:%Y-%m-%d %H:%M UTC} · Cached for up to 10 minutes"
        )
        st.subheader("Air temperature")
        st.altair_chart(forecast_chart(frame, "temperature_2m", "Temperature (°F)", tz, "#167a82"),
                        use_container_width=True)
        st.subheader("Chance of precipitation")
        st.altair_chart(forecast_chart(frame, "precipitation_probability",
                                       "Precipitation probability (%)", tz, "#416ec3", bars=True),
                        use_container_width=True)
        st.caption("Source: [Open-Meteo Weather Forecast API](https://open-meteo.com/en/docs)")
        return
    #成功的时候展示
    if st.button("Retry"):
        st.rerun()


if __name__ == "__main__":
    main()
