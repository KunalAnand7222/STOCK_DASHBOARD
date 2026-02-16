import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import io
import time

st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}
.metric-card {
background: rgba(255,255,255,0.05);
padding:20px;
border-radius:15px;
text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("Institutional Portfolio Risk Analytics Platform")

st.sidebar.header("Configuration")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.subheader("Portfolio Weights")

w1 = st.sidebar.slider("Reliance",0.0,1.0,0.30)
w2 = st.sidebar.slider("TCS",0.0,1.0,0.25)
w3 = st.sidebar.slider("HDFC Bank",0.0,1.0,0.20)
w4 = st.sidebar.slider("Infosys",0.0,1.0,0.15)
w5 = st.sidebar.slider("ICICI Bank",0.0,1.0,0.10)

weights = np.array([w1,w2,w3,w4,w5])
weights = weights/weights.sum()

tickers = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","^NSEI"]

data = yf.download(tickers,start=start_date,end=end_date)

close = data["Close"].dropna()
close.columns=["Reliance","TCS","HDFC Bank","Infosys","ICICI Bank","NIFTY50"]

returns = close.pct_change().dropna()
returns["Portfolio"]=returns.iloc[:,:5].dot(weights)

portfolio_cum=(1+returns["Portfolio"]).cumprod()
benchmark_cum=(1+returns["NIFTY50"]).cumprod()

annual_return=returns["Portfolio"].mean()*252
volatility=returns["Portfolio"].std()*np.sqrt(252)
beta=np.cov(returns["Portfolio"],returns["NIFTY50"])[0][1]/np.var(returns["NIFTY50"])
sharpe=(annual_return-0.06)/volatility
tracking_error=(returns["Portfolio"]-returns["NIFTY50"]).std()*np.sqrt(252)
max_drawdown=((portfolio_cum-portfolio_cum.cummax())/portfolio_cum.cummax()).min()

k1,k2,k3,k4,k5=st.columns(5)

def animated_metric(container,title,value):
    placeholder=container.empty()
    for i in np.linspace(0,value,40):
        placeholder.markdown(f"<div class='metric-card'><h4>{title}</h4><h2>{i:.2f}</h2></div>",unsafe_allow_html=True)
        time.sleep(0.01)
    placeholder.markdown(f"<div class='metric-card'><h4>{title}</h4><h2>{value:.2f}</h2></div>",unsafe_allow_html=True)

animated_metric(k1,"Annual Return %",annual_return*100)
animated_metric(k2,"Volatility %",volatility*100)
animated_metric(k3,"Sharpe Ratio",sharpe)
animated_metric(k4,"Beta",beta)
animated_metric(k5,"Max Drawdown %",max_drawdown*100)

tabs=st.tabs(["Performance","Risk","Simulation","Report"])

with tabs[0]:

    fig1=go.Figure()
    fig1.add_trace(go.Scatter(x=portfolio_cum.index,y=portfolio_cum,line=dict(color="#00BFFF",width=3),name="Portfolio"))
    fig1.add_trace(go.Scatter(x=benchmark_cum.index,y=benchmark_cum,line=dict(color="#FF4C4C",width=3),name="Benchmark"))
    fig1.update_layout(template="plotly_dark",xaxis_showgrid=False,yaxis_showgrid=False,height=500)
    st.plotly_chart(fig1,use_container_width=True)

    pie=go.Figure(data=[go.Pie(labels=close.columns[:5],values=weights,hole=.4)])
    pie.update_traces(marker=dict(colors=["#FFD700","#ADFF2F","#FF69B4","#FF8C00","#00CED1"]))
    pie.update_layout(template="plotly_dark",height=500)
    st.plotly_chart(pie,use_container_width=True)

with tabs[1]:

    rolling_vol=returns["Portfolio"].rolling(30).std()*np.sqrt(252)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=rolling_vol.index,y=rolling_vol,line=dict(color="#FFA500",width=3)))
    fig2.update_layout(template="plotly_dark",xaxis_showgrid=False,yaxis_showgrid=False,height=450)
    st.plotly_chart(fig2,use_container_width=True)

    rolling_beta=(returns["Portfolio"].rolling(30).cov(returns["NIFTY50"])/
                  returns["NIFTY50"].rolling(30).var())
    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=rolling_beta.index,y=rolling_beta,line=dict(color="#32CD32",width=3)))
    fig3.update_layout(template="plotly_dark",xaxis_showgrid=False,yaxis_showgrid=False,height=450)
    st.plotly_chart(fig3,use_container_width=True)

with tabs[2]:

    st.subheader("Live Rebalancing Simulation")
    rebalance_period=st.slider("Rebalance every N days",10,120,30)

    sim_returns=returns.iloc[:,:5]
    sim_portfolio=[]
    temp_weights=weights.copy()

    for i in range(len(sim_returns)):
        if i%rebalance_period==0:
            temp_weights=weights.copy()
        daily_return=np.dot(sim_returns.iloc[i],temp_weights)
        sim_portfolio.append(daily_return)

    sim_portfolio=pd.Series(sim_portfolio,index=sim_returns.index)
    sim_cum=(1+sim_portfolio).cumprod()

    fig4=go.Figure()
    fig4.add_trace(go.Scatter(x=sim_cum.index,y=sim_cum,line=dict(color="#FF1493",width=3)))
    fig4.update_layout(template="plotly_dark",xaxis_showgrid=False,yaxis_showgrid=False,height=500)
    st.plotly_chart(fig4,use_container_width=True)

    st.subheader("Monte Carlo Simulation")

    simulations=300
    days=252
    mc_results=np.zeros((days,simulations))

    for i in range(simulations):
        rand_returns=np.random.normal(returns["Portfolio"].mean(),
                                      returns["Portfolio"].std(),days)
        mc_results[:,i]=np.cumprod(1+rand_returns)

    fig5=go.Figure()
    for i in range(50):
        fig5.add_trace(go.Scatter(y=mc_results[:,i],mode="lines",
                                  line=dict(width=1),opacity=0.3))
    fig5.update_layout(template="plotly_dark",xaxis_showgrid=False,yaxis_showgrid=False,height=500)
    st.plotly_chart(fig5,use_container_width=True)

with tabs[3]:

    st.subheader("Generate Portfolio Report")

    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=12)
    pdf.cell(200,10,txt="Portfolio Risk Report",ln=True)
    pdf.cell(200,10,txt=f"Annual Return: {annual_return:.2%}",ln=True)
    pdf.cell(200,10,txt=f"Volatility: {volatility:.2%}",ln=True)
    pdf.cell(200,10,txt=f"Sharpe Ratio: {sharpe:.2f}",ln=True)
    pdf.cell(200,10,txt=f"Beta: {beta:.2f}",ln=True)
    pdf.cell(200,10,txt=f"Tracking Error: {tracking_error:.2f}",ln=True)

    pdf_output=pdf.output(dest='S').encode('latin1')
    st.download_button("Download PDF Report",
                       data=pdf_output,
                       file_name="Portfolio_Report.pdf",
                       mime="application/pdf")
