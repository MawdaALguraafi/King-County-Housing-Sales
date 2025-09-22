import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re, pickle, statsmodels.api as sm

st.set_page_config(page_title="House Sales in King County Dashboard", layout="wide")

# chart colors
COLORS = ["#00C767", "#0B3D0B"]  # rich green + dark green (not neon)
PIE_COLORS = ["#1B5E20", "#2E7D32", "#66BB6A", "#A5D6A7"]  # softer greens
BG_PAPER = "rgba(0,0,0,0)"
BG_PLOT  = "rgba(0,0,0,0)"
FONT_COL = "#FFFFFF"
GRID_COL = "rgba(102,187,106,0.35)"  # soft green grid

def apply_dark_layout(fig, show_legend=True):
    fig.update_layout(
        template="simple_white",
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(color=FONT_COL),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if show_legend else dict(),
        margin=dict(t=60, r=20, b=40, l=60)
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COL, zeroline=False, linecolor="#2E7D32")
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COL, zeroline=False, linecolor="#2E7D32")
    return fig

st.markdown("""
<style>
.stApp, [data-testid="stSidebar"] { background-color: #000000; color: #FFFFFF; }
a { color: #66BB6A !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(1) { background-color: rgba(11,61,11,0.25) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div [class*="inner"] { background-color: #0B3D0B !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #0B3D0B !important; border: 2px solid #0B3D0B !important; box-shadow: 0 0 0 4px rgba(11,61,11,0.25) !important;
}
[data-testid="stMetric"] { background: transparent; border: none; padding: 14px 16px; }
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# load data
df = pd.read_csv("/content/cleaned_file.csv")
date_col = "sale_date" if "sale_date" in df.columns else "date"
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.rename(columns={date_col: "sale_date"})
df = df.dropna(subset=["sale_date", "price"]).copy()
df["sale_ym"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()

# extra columns
if "num_bedrooms" in df.columns:
    df["num_bedrooms_int"] = pd.to_numeric(df["num_bedrooms"], errors="coerce").astype("Int64")
if "num_bathrooms" in df.columns:
    df["num_bathrooms_int"] = pd.to_numeric(df["num_bathrooms"], errors="coerce").round().astype("Int64")
if "waterfront" in df.columns:
    df["waterfront_int"] = pd.to_numeric(df["waterfront"], errors="coerce").fillna(0).astype(int)
    df["waterfront_label"] = df["waterfront_int"].map({1: "Waterfront", 0: "Non-waterfront"})
else:
    df["waterfront_label"] = "Unknown"

# title
st.title("King County Housing Sales Dashboard")

with st.expander("Dataset Description"):
    st.markdown("""
This dataset contains detailed information about house sales in King County.
It includes the following fields:

- **Sale date**: The date when the property was sold.
- **Price**: The sale price of the property.
- **Bedrooms and bathrooms**: The number of bedrooms and bathrooms in the house.
- **Living and lot square footage**: The interior living space and the total lot size.
- **Floors, waterfront, and view**: Structural details, whether the property is on the waterfront, and the quality of the view.
- **Market trend**: Used for analyzing how prices and demand have changed over time.
""")

# filters
st.sidebar.header("Filters")
ym_all = sorted(df["sale_ym"].dropna().unique().tolist())
ym_all = [pd.Timestamp(t).to_pydatetime() for t in ym_all] or [pd.Timestamp("2014-01-01"), pd.Timestamp("2015-12-01")]
min_ym, max_ym = min(ym_all), max(ym_all)
ym_range = st.sidebar.slider("Sale Date (YM)", min_value=min_ym, max_value=max_ym, value=(min_ym, max_ym), format="YYYY-MM")

# price filter
min_price, max_price = int(df["price"].min()), int(df["price"].max())
price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=10000
)

beds = ["All"] + (sorted(df["num_bedrooms_int"].dropna().unique().tolist()) if "num_bedrooms_int" in df.columns else [])
baths = ["All"] + (sorted(df["num_bathrooms_int"].dropna().unique().tolist()) if "num_bathrooms_int" in df.columns else [])
sel_beds  = st.sidebar.selectbox("Bedrooms", beds)
sel_baths = st.sidebar.selectbox("Bathrooms", baths)

# apply filters
f = df[
    (df["sale_ym"] >= pd.Timestamp(ym_range[0])) &
    (df["sale_ym"] <= pd.Timestamp(ym_range[1])) &
    (df["price"] >= price_range[0]) &
    (df["price"] <= price_range[1])
].copy()

if sel_beds  != "All" and "num_bedrooms_int" in f.columns:
    f = f[f["num_bedrooms_int"] == int(sel_beds)]
if sel_baths != "All" and "num_bathrooms_int" in f.columns:
    f = f[f["num_bathrooms_int"] == int(sel_baths)]

def human_format(num):
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"${human_format(f['price'].sum())}")
c2.metric("Avg Price",    f"${human_format(f['price'].mean() if len(f) else 0)}")
c3.metric("Median Price", f"${human_format(f['price'].median() if len(f) else 0)}")
c4.metric("Homes Sold",   human_format(len(f)))


# row 1: line
r1c1 = st.columns(1)[0]
with r1c1:
    st.subheader("Monthly Average Sale Price")
    monthly = f.groupby("sale_ym", as_index=False)["price"].mean().sort_values("sale_ym")
    if len(monthly):
        fig = px.line(monthly, x="sale_ym", y="price", markers=True, color_discrete_sequence=[COLORS[0]])
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig.update_layout(xaxis_title="Year-Month", yaxis_title="Avg Price ($)", hovermode="x unified")
        fig.update_yaxes(tickformat=",.0f")
        st.plotly_chart(apply_dark_layout(fig, show_legend=False), use_container_width=True)

# row 2: scatter + bar
r2c1, r2c2 = st.columns(2)
with r2c1:
    st.subheader("Price vs Living Sqft ")
    if {"living_sqft","price"}.issubset(f.columns):
        sc = f.dropna(subset=["living_sqft","price"])
        if len(sc):
            fig = px.scatter(sc, x="living_sqft", y="price", opacity=0.55, color_discrete_sequence=[COLORS[1]])
            z = np.polyfit(sc["living_sqft"], sc["price"], 2)
            p = np.poly1d(z)
            x_sorted = np.linspace(sc["living_sqft"].min(), sc["living_sqft"].max(), 200)
            fig.add_scatter(x=x_sorted, y=p(x_sorted), mode="lines", line=dict(color=COLORS[0], width=3), name="Poly Fit")
            fig.update_layout(xaxis_title="Living Sqft", yaxis_title="Price ($)")
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(apply_dark_layout(fig), use_container_width=True)

with r2c2:
    st.subheader("Avg Price by Bedrooms & Bathrooms")
    bars = []
    if "num_bedrooms_int" in f.columns:
        b = f.dropna(subset=["price","num_bedrooms_int"]).groupby("num_bedrooms_int", as_index=False)["price"].mean()
        if len(b):
            b["Category"] = "Bedrooms"; b["Value"] = b["num_bedrooms_int"].astype(str); bars.append(b[["Category","Value","price"]])
    if "num_bathrooms_int" in f.columns:
        t = f.dropna(subset=["price","num_bathrooms_int"]).groupby("num_bathrooms_int", as_index=False)["price"].mean()
        if len(t):
            t["Category"] = "Bathrooms"; t["Value"] = t["num_bathrooms_int"].astype(str); bars.append(t[["Category","Value","price"]])
    if bars:
        df_bar = pd.concat(bars)
        fig_bar = px.bar(df_bar, x="Value", y="price", color="Category", barmode="group", text_auto=".2s",
                         color_discrete_sequence=COLORS)
        fig_bar.update_layout(xaxis_title="Count", yaxis_title="Avg Price ($)")
        fig_bar.update_yaxes(tickformat=",.0f")
        st.plotly_chart(apply_dark_layout(fig_bar), use_container_width=True)

# row 3: bars + pie
r3c1, r3c2 = st.columns(2)
with r3c1:
    st.subheader("Average Price by Waterfront")
    if "waterfront_label" in f.columns and f["waterfront_label"].nunique() > 1:
        wf = f.dropna(subset=["price"]).groupby("waterfront_label", as_index=False)["price"].mean()
        if len(wf):
            fig = px.bar(
                wf, x="waterfront_label", y="price", text_auto=".2s",
                color="waterfront_label", color_discrete_sequence=COLORS
            )
            fig.update_layout(
                xaxis_title="Waterfront",
                yaxis_title="Avg Price ($)",
                xaxis=dict(title="", tickmode="array", tickvals=wf["waterfront_label"]),
                bargap=0.4,
                uniformtext_minsize=12,
                uniformtext_mode="hide"
            )
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(apply_dark_layout(fig, show_legend=False), use_container_width=True)

with r3c2:
    st.subheader("Average Price Share by View")
    if "view" in f.columns:
        vg = f.dropna(subset=["view","price"]).groupby("view", as_index=False)["price"].mean().sort_values("view")
        if len(vg):
            fig = px.pie(vg, names="view", values="price", hole=0.3, color_discrete_sequence=PIE_COLORS)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(apply_dark_layout(fig), use_container_width=True)

# row 4: preview
r4c1 = st.columns(1)[0]
with r4c1:
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

st.subheader("Statistics Summary")
summary_cols = ["sale_date", "price", "num_bedrooms", "num_bathrooms", "living_sqft", "year_built"]
available_cols = [c for c in summary_cols if c in f.columns]
if not f.empty and available_cols:
    st.write(f[available_cols].describe(include="all"))
else:
    st.info("No data available for the selected filters.")

st.markdown("---")
st.markdown("**Data Source:** House Sales in King County, USA (from Kaggle)")


# 💬 House Price Prediction Chatbot

st.header("💬 House Price Prediction Chatbot")
with st.expander("How to use the chatbot (click to expand)"):
    st.markdown("""
Enter house details in one message. The chatbot will predict the price.

Example: `living=2000 grade=8 bath=2 bed=3 cond=4 wf=0 year=2015 month=6`
""")

@st.cache_resource
def load_artifacts():
    with open("ols_model.pkl", "rb") as f:
        ols_model = pickle.load(f)
    with open("trend_train.pkl", "rb") as f:
        trend_train = pickle.load(f)  # dict-like {Period('YYYY-MM','M'): trend_value}
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)  # expects keys: features, X_train_columns, last_trend
    return ols_model, trend_train, meta

ols_model, trend_train, meta = load_artifacts()
last_trend   = meta.get("last_trend", 0.0)
features     = list(meta.get("features", []))
X_train_cols = list(meta.get("X_train_columns", []))

def predict_price(living_sqft, grade, num_bathrooms, num_bedrooms,
                  condition, waterfront, year, month):

    row = {
        "living_sqft": living_sqft,
        "grade": grade,
        "num_bathrooms": num_bathrooms,
        "num_bedrooms": num_bedrooms,
        "condition": condition,
        "waterfront": waterfront,
    }


    if "year" in features:
        row["year"] = year
    if "month" in features:
        row["month"] = month

    df_row = pd.DataFrame([row]).astype(float)


    ym = pd.Period(f"{int(year)}-{int(month):02d}", freq="M")
    mt = trend_train.get(ym, last_trend) if hasattr(trend_train, "get") else last_trend
    df_row["market_trend"] = float(mt)

    feature_list = features if len(features) else list(df_row.columns)

    X_no_const = df_row.reindex(columns=feature_list, fill_value=0.0).astype(float)


    X = sm.add_constant(X_no_const, has_constant="add")
    X = X.reindex(columns=X_train_cols, fill_value=0.0)


    y_log = float(ols_model.predict(X).iloc[0])
    smear = float(np.exp(getattr(ols_model, "mse_resid", 0) / 2.0))
    return round(float(np.exp(y_log) * smear), 2)

def extract_float(key, text, default=None):
    m = re.search(rf"{key}\s*=?\s*([-+]?\d*\.?\d+)", text, re.IGNORECASE)
    return float(m.group(1)) if m else default

def extract_int(key, text, default=None):
    val = extract_float(key, text, None)
    return int(val) if val is not None else default


if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Hi! Enter house details (living, grade, bath, bed, cond, wf, year, month) and I’ll predict the price."}
    ]

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

msg = st.chat_input("Type here, e.g. living=5000 grade=3 bath=3 bed=3 cond=3 wf=1 year=2026 month=4")
if msg:
    st.session_state.chat.append({"role": "user", "content": msg})
    with st.chat_message("user"):
        st.write(msg)

    living_sqft   = extract_float("living", msg, None)
    grade         = extract_int("grade", msg, None)
    num_bathrooms = extract_int("bath", msg, None)
    num_bedrooms  = extract_int("bed", msg, None)
    condition     = extract_int("cond", msg, None)
    waterfront    = extract_int("wf", msg, None)
    year          = extract_int("year", msg, None)
    month         = extract_int("month", msg, None)

    values = {
        "living_sqft": living_sqft, "grade": grade, "num_bathrooms": num_bathrooms,
        "num_bedrooms": num_bedrooms, "condition": condition, "waterfront": waterfront,
        "year": year, "month": month
    }
    missing = [k for k,v in values.items() if v is None]

    if missing:
        reply = "Missing fields: **" + ", ".join(missing) + "**"
    else:
        try:
            pred = predict_price(**values)
            reply = f"Predicted Price: **${pred:,.2f}**"
        except Exception as e:
            reply = f"Error: {e}"

    st.session_state.chat.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
