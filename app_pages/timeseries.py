import json

import pandas as pd
import streamlit as st

from agentic_ds import timeseries as ts
from agentic_ds.ui_helpers import metric_row, narrative_button

st.title("Time series")
st.caption("Stationarity, decomposition, autocorrelation, and a small ARIMA fit + forecast.")

with st.container(border=True):
    if st.button("Load a synthetic monthly series (demo)", icon=":material/lightbulb:"):
        st.session_state.ts_series = ts.synthetic_monthly_series()
        st.session_state.ts_source_label = "synthetic: monthly series (trend + yearly seasonality + noise — not real data)"
        st.session_state.ts_arima_result = None
        st.rerun()

    st.caption(
        "None of this app's other built-in datasets (iris/wine/breast_cancer/diabetes/titanic) have a time axis, "
        "so this demo series is honestly synthetic rather than a mislabeled real one. Or use any numeric column "
        "from your own loaded dataset below."
    )

    df = st.session_state.df
    if df is not None:
        with st.expander("Or build a series from the currently loaded dataset"):
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                value_col = st.selectbox("Value column", numeric_cols)
                date_col = st.selectbox("Date/order column", ["(row order)"] + all_cols)
                if st.button("Use this column as a time series", icon=":material/timeline:"):
                    if date_col == "(row order)":
                        series = df[value_col].reset_index(drop=True)
                    else:
                        series = df.dropna(subset=[date_col, value_col]).set_index(date_col)[value_col].sort_index()
                    series.name = value_col
                    st.session_state.ts_series = series
                    st.session_state.ts_source_label = f"{st.session_state.source_label} · {value_col}"
                    st.session_state.ts_arima_result = None
                    st.rerun()
            else:
                st.caption("No numeric columns in the currently loaded dataset.")

series = st.session_state.get("ts_series")
if series is None:
    st.info("Load the synthetic demo series above, or build one from a loaded dataset.")
    st.stop()

st.caption(f"Series: {st.session_state.ts_source_label} · {len(series)} points")
st.line_chart(series)

with st.container(border=True):
    st.markdown("**Stationarity**")
    stationarity = ts.check_stationarity(series)
    if stationarity["verdict"] == "non-stationary":
        st.warning(ts.interpret_stationarity(stationarity))
    else:
        st.write(ts.interpret_stationarity(stationarity))
    with st.expander("ADF and KPSS details"):
        st.json(stationarity)

    suggested_d, _ = ts.suggest_differencing_order(series)
    st.caption(f"Suggested differencing order (d) to reach stationarity: **{suggested_d}**")

with st.container(border=True):
    st.markdown("**Seasonal decomposition**")
    period = st.number_input("Period (e.g. 12 for monthly data with yearly seasonality)", min_value=2, value=12)
    if len(series) >= 2 * period:
        components = ts.decompose(series, period=period)
        st.line_chart(components[["trend", "seasonal"]])
        with st.expander("Residual"):
            st.line_chart(components[["residual"]])
    else:
        st.caption(f"Need at least {2 * period} points for period={period}.")

with st.container(border=True):
    st.markdown("**Autocorrelation**")
    acf_pacf = ts.autocorrelation(series)
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("ACF")
        st.bar_chart(acf_pacf.set_index("lag")["acf"])
    with col_b:
        st.caption("PACF")
        st.bar_chart(acf_pacf.set_index("lag")["pacf"])

    whiteness = ts.whiteness_test(series)
    caption = f"Ljung-Box on the raw series: p = {whiteness['p_value']} → " + (
        "no significant autocorrelation detected (little for ARIMA to model)."
        if whiteness["white_noise"]
        else "significant autocorrelation present — there's real structure to model."
    )
    st.caption(caption)

with st.container(border=True):
    st.markdown("**ARIMA fit + forecast**")
    c1, c2, c3, c4 = st.columns(4)
    p = c1.number_input("p (AR order)", min_value=0, max_value=5, value=1)
    d = c2.number_input("d (differencing)", min_value=0, max_value=2, value=int(suggested_d))
    q = c3.number_input("q (MA order)", min_value=0, max_value=5, value=1)
    steps = c4.number_input("Forecast steps", min_value=1, max_value=36, value=12)

    if st.button("Fit ARIMA", icon=":material/play_arrow:"):
        try:
            st.session_state.ts_arima_result = ts.fit_arima(series, order=(p, d, q), forecast_steps=steps)
        except Exception as e:
            st.error(f"Fit failed: {e}")

    arima_result = st.session_state.get("ts_arima_result")
    if arima_result:
        if arima_result["residual_whiteness"]["white_noise"]:
            st.write(ts.interpret_arima(arima_result))
        else:
            st.warning(ts.interpret_arima(arima_result))

        metric_row({"aic": arima_result["aic"], "bic": arima_result["bic"], "n_obs": arima_result["n_obs"]})

        st.caption("Coefficients")
        st.dataframe(arima_result["coefficients"], width="stretch", hide_index=True)

        forecast_df = pd.DataFrame(arima_result["forecast"])
        st.caption("Forecast (mean + 95% CI)")
        st.dataframe(forecast_df, width="stretch", hide_index=True)

        narrative_button(
            "You are a statistics teacher. Given a fitted ARIMA model's summary "
            "(coefficients, fit stats, residual diagnostics, forecast) as JSON, explain "
            "in 3-5 plain-English sentences: whether the model fits well, what the "
            "forecast trend looks like, and any caveats. No jargon.",
            json.dumps(arima_result, indent=2, default=str),
        )
