import yfinance as yf
import pandas as pd
import calendar
import datetime
# import mplfinance as mpf
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from darts.timeseries import TimeSeries
from darts.models import Theta
import pickle


st.set_page_config(layout="wide",
#  theme={
#         "primaryColor": "#1f77b4",
#         "backgroundColor": "#0e1117",
#         "secondaryBackgroundColor": "#262730",
#         "textColor": "#fafafa",
#         "font": "sans serif"
#     }
    initial_sidebar_state = 'auto',
    page_icon="🧊"
)
st.title('HDFC Bank Mututal Fund')
stock_name = st.sidebar.selectbox("Select the Mutual Fund",("HDFC Bank","Gili Gili"))

with st.sidebar:
    Page = st.radio("",('Description','Candlestick','Volume Distribution','Adjusted Closing Price','Prediction'))



mapping_mf = {'HDFC Bank':'HDFCBANK.NS'}

comp_ticket = mapping_mf[stock_name]
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5*365)  # Last 5 years

# Download the stock data
stock_data = yf.download(comp_ticket, start=start_date, end=end_date)


df = pd.DataFrame(stock_data)
# df.index = df.index.strftime('%Y-%m-%d')
df['Year_Month'] = df.index.to_period('M')
df['Year'] = df.index.to_period('Y')
df['Month'] = df.index.month
df['Day'] = df.index.day

def desc_func():
    # st.image('stock_data.jpg')
    st.subheader("Glimpse of the Dataset")
    st.dataframe(df.drop(['Year_Month','Year','Month','Day'],axis = 1).tail(7).round(2),use_container_width=False)



###################################################################################################################
###################################################################################################################

def plot_candlestick(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)

    # Add candlestick trace to first subplot
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Candlestick'),
                row=1, col=1)

    # Add volume bars trace to second subplot
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'],
                        marker_color='lightblue',
                        text = data['Volume'],
                        textposition='outside',
                        name='Volume'),
                row=2, col=1)

    # Update x-axis properties
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    # Update y-axis properties
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=False,template='plotly_dark')
    return fig

def cndl_stck():
    st.subheader('CandleStick Chart with Volume Distribution')
    last_date = pd.to_datetime(df.index[len(df)-1])
    dt_period = st.selectbox('Select the time period',('last 7 days','last 1 month','last 2 months','last 3 months','last 6 months', 'last 1 year','custom'))
    if dt_period == 'last 7 days':
        first_date = last_date - pd.DateOffset(days = 7)
    elif dt_period == 'last 1 month':
        first_date = last_date - pd.DateOffset(months = 1)
    elif dt_period =='last 2 months':
        first_date =last_date - pd.DateOffset(months = 2)
    elif dt_period =='last 3 months':
        first_date =last_date - pd.DateOffset(months = 3)
    elif dt_period =='last 6 months':
        first_date =last_date - pd.DateOffset(months = 6)
    elif dt_period =='last 1 year':
        first_date =last_date - pd.DateOffset(years = 1)
    else:
        prev_date =last_date - pd.DateOffset(months = 3)
        first_date = last_date - pd.DateOffset(years = 5)
        first_date,last_date = st.slider('Select the window for more detailed view', first_date.to_pydatetime(),last_date.to_pydatetime(),(prev_date.to_pydatetime(),last_date.to_pydatetime()))

    data = df[df.index.isin(pd.date_range(first_date,last_date))]
    data = pd.date_range(first_date,last_date).to_frame().drop([0],axis = 1).merge(data,right_index=True,left_index=True,how = 'left').ffill()
    
    return st.plotly_chart(plot_candlestick(data))





###################################################################################################################
###################################################################################################################
LDT = pd.to_datetime(df.index[len(df)-1])
FDT = LDT - pd.DateOffset(years = 5)

fil_df = df[df.index.isin(pd.date_range(FDT,LDT))][['Volume','Year','Year_Month','Month','Day']]
yr_df = fil_df.groupby('Year')['Volume'].sum().to_frame().reset_index()
yr_df['Year'] = yr_df['Year'].astype(str)

ym_df = fil_df.groupby('Year_Month')['Volume'].sum().to_frame().reset_index()
ym_df['Year_Month'] = ym_df['Year_Month'].dt.to_timestamp()

mon_df = fil_df.groupby('Month')['Volume'].mean().to_frame().reset_index()
mon_df['Month'] = mon_df['Month'].apply(lambda x: calendar.month_name[x])

day_df = fil_df.groupby('Day')['Volume'].mean().to_frame().reset_index().sort_values(by = 'Day')
# mon_df['Day'] = mon_df['Day'].dt.to_timestamp()


def vol_dist():
    st.subheader('Yearly Volume Distribution')
    st.bar_chart(yr_df,x = 'Year',y = 'Volume')

    st.subheader('Monthly Volume Distribution across Year')
    st.bar_chart(ym_df,x = 'Year_Month',y = 'Volume')

    st.subheader('Monthly Volume Distribution')
    st.bar_chart(mon_df,x = 'Month',y = 'Volume')

    st.subheader('Daily Volume Distribution')
    st.bar_chart(day_df,x = 'Day',y = 'Volume')



###################################################################################################################
###################################################################################################################

def adj_cl():
    st.subheader('Adj. Closing Price Distribution')
    lt_dt = st.date_input('end date');ft_dt = st.date_input('start date',lt_dt - pd.DateOffset(months = 6))
    cl_data  = df[df.index.isin(pd.date_range(ft_dt,lt_dt))][['Adj Close']]
    cl_data.index = pd.to_datetime(cl_data.index)
    st.line_chart(cl_data)


###################################################################################################################
###################################################################################################################
def Model():
    if Page == 'Prediction':
        st.subheader('Prediction for the Forecast Horizon')
        tr_bool = st.selectbox('Select whether the Model need to be trained/not',(False,True))
        pred_horizon = st.number_input('Prediction Horizon',min_value=0, max_value=3650, step=1, value=150, format="%d")
        date = str(pd.to_datetime(datetime.datetime.timestamp(datetime.datetime.now()),unit = 's') - pd.offsets.MonthBegin(1))[:10]
        if tr_bool:
            cl_data  = df[df.index.isin(pd.date_range('2021-01-01',date))][['Adj Close']]
            cl_df = pd.date_range('2021-01-01',date).to_frame().merge(cl_data,right_index=True,left_index=True,how = 'left').drop(0,axis = 1).ffill()
            cl_df['Adj Close'] = cl_df['Adj Close'].astype('float32')
            series = TimeSeries.from_dataframe(cl_df)
            model = Theta(theta = 0.5,seasonality_period = 365)
            model.fit(series)
            prediction = model.predict(pred_horizon)
            with open(f'./model_object/{date}.pkl','wb') as file:
                pickle.dump(model,file)
        else:
            try:
                with open(f'./model_object/{date}.pkl','rb') as file:
                    loaded_model = pickle.load(file)
                    prediction = loaded_model.predict(pred_horizon)
            except:
                st.write('**No model object Found. Train the model first!!!!**')
        try:
            st.line_chart(prediction.pd_dataframe())
        except:
            st.write('**Either Model is not trained or prediction horizon is 0**')

        st.subheader('Key Summary')
        PD = prediction.pd_dataframe()
        PD['Month'] = PD.index.month
        PD['Year'] = PD.index.year.astype(str)
        PD['Year_Month'] = pd.to_datetime(PD.index) - pd.offsets.MonthBegin(1)

        if PD.Year.unique().shape[0]>1:
            final_df = PD.groupby('Year')['Adj Close'].mean().reset_index()
        else:
            if PD.Month.unique().shape[0] > 1:
                final_df = PD.groupby('Month')['Adj Close'].mean().reset_index()
            else:
                final_df = PD[['Adj Close']]

        final_df['change'] = (final_df['Adj Close']/final_df['Adj Close'].shift(1) - 1)
        final_df.loc[0,'change'] = final_df['Adj Close'][0]/df.iloc[len(df)-1,4]  - 1
        st.dataframe(final_df)
        st.write("Today's adj. closing price : ", df.iloc[len(df)-1,4])
            




if Page == 'Description':
    desc_func()
elif Page == 'Candlestick':
    cndl_stck()
elif Page == 'Volume Distribution':
    vol_dist()
elif Page == 'Adjusted Closing Price':
    adj_cl()
elif Page == 'Prediction':
    Model()
else:
    Model()






