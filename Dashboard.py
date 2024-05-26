import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def dashboard():
    dff = pd.read_csv(r"Data/Data_annually.csv")
    st.title("AI & Advancing Future Sustainability")
    year_filter = st.selectbox("Select the Year", pd.unique(dff['Year']))

    df = dff[dff['Year'] == year_filter]

    mean_temp = float(df['T'])
    max_temp = float(df['TM'])
    min_temp = float(df['Tm'])

    mean_precip = float(df['PP'])
    V = float(df['V'])
    SN = float(df['SN'])

    st.markdown("### Statistics for Selected Year")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Mean Temperature**")
        st.info(f"{mean_temp:.2f} °C")
        st.markdown("**Precipitation**")
        st.info(f"{mean_precip:.2f} mm")

    with col2:
        st.markdown("**Maximum Temperature**")
        st.success(f"{max_temp:.2f} °C")
        st.markdown("**Wind Speed**")
        st.success(f"{V:.2f} Km")

    with col3:
        st.markdown("**Minimum Temperature**")
        st.error(f"{min_temp:.2f} °C")
        st.markdown("**SN**")
        st.error(f"{SN:.2f}")

    placeholder = st.empty()

    with placeholder.container():
        fig_col1, fig_col2, fig_col3 = st.columns(3)

        with fig_col1:
            st.markdown("### Heatmap of PP vs TM")
            fig = px.density_heatmap(data_frame=dff, y='PP', x='TM')
            st.write(fig)

        with fig_col2:
            st.markdown("### Histogram of PP")
            fig2 = px.histogram(data_frame=dff, x='PP')
            st.write(fig2)

        with fig_col3:
            st.markdown("### Histogram of TM")
            fig3 = px.histogram(data_frame=dff, x='TM')
            st.write(fig3)

        st.markdown("### Trend Charts")
        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:
            st.markdown("#### Mean Temperature per Year")
            fig4 = px.line(dff, x='Year', y='T')
            st.write(fig4)

        with trend_col2:
            st.markdown("#### Mean Precipitation per Year")
            fig5 = px.line(dff, x='Year', y='PP')
            st.write(fig5)

        mt_col1, tab_col2 = st.columns(2)
        with mt_col1:
            st.markdown("### Correlation Matrix")
            corr = dff.corr()
            fig_corr = px.imshow(corr, text_auto=True, width=700, height=700)
            st.write(fig_corr)

        with tab_col2:
            st.markdown("### Detailed Data View")
            st.dataframe(dff)