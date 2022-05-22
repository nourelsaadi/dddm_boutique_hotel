#importing packages
from pickle import NONE
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.tools as tools 
import warnings
warnings.filterwarnings('ignore')

init_notebook_mode(connected = "true")

#Making sure the App is first loaded with a wide layout 
st.set_page_config(
     layout="wide")

#Setting the horizontal navigation bar
choose = option_menu(menu_title="",
                    options= ["Home", "Dashboard", "Will They Cancel?"],
                    icons=["bi bi-house-door", "graph-up","people"],
                    menu_icon= NONE,
                    default_index=0,
                    orientation="horizontal")

#Setting a place to upload the dataset
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

# Setting the Home Page
if choose == "Home":
     st.markdown("<h1 style='text-align: center; color: #e76e58;'>The Boutique Hotel</h1><br>", unsafe_allow_html=True) 
     #Introductory Statements
     st.markdown("<center><h5> Summer is around the corner and you are the most popular Boutique Hotel in the area! <br> With your two branches and a growing amount of sudden cancellations, you are in need of an App that would help you predict if a specific customer will cancel their booking or not & act accordingly. <br> Upload the dataset extracted from your system and let our Streamlit App help you. </center></h5>", unsafe_allow_html=True)
     
#Setting the Dashboard-like Page
if choose == "Dashboard":
    #Conditionning the page (if data is uploaded)
     if uploaded_file is not None:
         #Creating a dataframe from the uploaded dataset
         df1 = pd.read_csv(uploaded_file)
         #Simple preparation to the dataset (as per findings in ML assignment 3, but all the pre-processing made here is simplified and further basic)ss
         df1.drop(columns = ['company'],inplace= True)
         df1['agent']= df1['agent'].fillna(0)
         df1['country'].fillna(df1['country'].mode()[0], inplace=True)
         df1['children']= df1['children'].fillna(0)
         
         #Dividing the page into a first row with several columns
         col0, col1, col2, col3 = st.columns( [0.165, 0.165, 0.33, 0.33])

         with col0:
             canceled= df1[df1['is_canceled']==1]
             not_canceled= df1[df1['is_canceled']==0]
             st.metric(label="Total Number Of Bookings", value=len(df1.index))  

         with col1:
             st.metric(label="Total Number of Cancellations", value=len(canceled.index))
         
         with col2:
             labels = ['Canceled','Not Canceled']
             values = [len(canceled.index), len(not_canceled.index)]
             fig = go.Figure(data=[go.Pie(labels=labels, values=values,  textinfo='label+percent',hole=.3)])
             fig.update_layout(title_text='How many of our Bookings were Cancelled?',
                  title_x=0.2, title_font=dict(size=22),height=400, width=550) 
             st.plotly_chart(fig, use_container_width=True)    

         with col3:
             feature = ["is_canceled", 'customer_type']
             data3 = pd.crosstab(df1[feature[0]], df1[feature[1]])
             data3['Cancelation'] = data3.index

             trace4 = go.Bar(
                x = data3['Cancelation'].index.values,
                y = data3.Contract,
                name='Contract')

             trace5 = go.Bar(
                x = data3['Cancelation'].index.values,
                y = data3.Group,
                name='Group')

             trace6 = go.Bar(
                x = data3['Cancelation'].index.values,
                y = data3.Transient,
                name='Transient')

             trace7 = go.Bar(
                x = data3['Cancelation'].index.values,
                y = data3['Transient-Party'],
                name='Transient-Party')

             fig = tools.make_subplots(rows=1, 
                          cols=1)
                          
             fig.append_trace(trace4, 1, 1)
             fig.append_trace(trace5, 1, 1)
             fig.append_trace(trace6, 1, 1)
             fig.append_trace(trace7, 1, 1)

             fig.update_layout(title_text='How Does the Customer type affect Bookings?',
                  title_x=0.2, title_font=dict(size=22),height=400, width=550) 
             st.plotly_chart(fig, use_container_width=True)    
          
         #Dividing the page into a first row with several columns
         col4, col5, col6 = st.columns( [0.33, 0.33, 0.33])
         with col4:
             country_freq = df1['country'].value_counts().to_frame()
             country_freq.columns = ['count']
             fig = px.choropleth(country_freq, color=np.log(country_freq['count']),
                    locations=country_freq.index,
                    hover_name=country_freq.index,
                    color_continuous_scale=px.colors.sequential.deep)
             fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
             fig.update_layout(title_text='Where do our Customers Come From?',
                  title_x=0.2, title_font=dict(size=22),height=400, width=550 )  # Location and the font size of the main title
             st.plotly_chart(fig, use_container_width=True)   

         with col6:
             feature = ["is_canceled", 'deposit_type']
             data4 = pd.crosstab(df1[feature[0]], df1[feature[1]])
             data4['Cancelation'] = data4.index

             trace3 = go.Bar(
             x = data4['Cancelation'].index.values,
             y = data4['No Deposit'],
             name='No Deposit')

             trace4 = go.Bar(
             x = data4['Cancelation'].index.values,
             y = data4['Non Refund'],
             name='Non Refund')

             trace5 = go.Bar(
             x = data4['Cancelation'].index.values,
             y = data4.Refundable,
             name='Refundable')

             fig = tools.make_subplots(rows=1, 
                          cols=1)

             fig.append_trace(trace3, 1, 1)
             fig.append_trace(trace4, 1, 1)
             fig.append_trace(trace5, 1, 1)
             fig.update_layout(title_text='How do Deposits affect Cancellations?',
                  title_x=0.2,height=400, width=550, boxmode='group',title_font=dict(size=22)) 
             st.plotly_chart(fig, use_container_width=True)   

         with col5:
             city= df1[df1['hotel']=='City Hotel']
             resort= df1[df1['hotel']=='Resort Hotel']
             
             labels = ['City Hotel','Resort Hotel']
             values = [len(city.index), len(resort.index)]
             # Use `hole` to create a donut-like pie chart
             fig = go.Figure(data=[go.Pie(labels=labels, values=values,  textinfo='label+percent',hole=.3)])
             fig.update_layout(title_text='Booking Distribution Across Our 2 Branches',
                  title_x=0.2, title_font=dict(size=22),height=400, width=550) 
             st.plotly_chart(fig, use_container_width=True)    
     
     else:
         # Show warning if dataset not uploaded yet
         st.warning("Please Upload a Dataset in the Data Upload Slot Above")

if choose == "Will They Cancel?":
     #Conditionning the page (if data is uploaded)
     if uploaded_file is not None:
         #the dataset was not carried across the if statement, even when using the turnarounds, so the same simple methods used above were repeated here
         #Creating a dataframe from the uploaded dataset
         df = pd.read_csv(uploaded_file)
         #Simple preparation to the dataset (as per findings in ML assignment 3, but all the pre-processing made here is simplified and further basic)ss
         df.drop(columns = ['company'],inplace= True)
         df['agent']= df['agent'].fillna(0)
         df['country'].fillna(df['country'].mode()[0], inplace=True)
         df['children']= df['children'].fillna(0)
         df1=df
        
         #Basic Preprocessing-- Manual Encoding: Custom mapping
         df1['hotel'] = df1['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
         df1['arrival_date_month'] = df1['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
         #Encoding using labelEncoder, no pipelines were created
         le = LabelEncoder()
         df1['deposit_type'] = le.fit_transform(df1['deposit_type'])
         df1['customer_type'] = le.fit_transform(df1['customer_type'])
         df1['market_segment'] = le.fit_transform(df1['market_segment'])
         df1['distribution_channel'] = le.fit_transform(df1['distribution_channel'])
         df1['reserved_room_type'] = le.fit_transform(df1['reserved_room_type'])
         df1['assigned_room_type']= le.fit_transform(df1['assigned_room_type'])
         df1['reservation_status_date'] = le.fit_transform(df1['reservation_status_date'])
         #according to analysis in assignment 3, the top 11 features that determine hotel cancellations were the ones in the 'chosen' list. These will thus be used to get the prediction we need
         chosen=['hotel','lead_time','adr','total_of_special_requests','booking_changes','market_segment','deposit_type','assigned_room_type','customer_type','required_car_parking_spaces','previous_cancellations']
         X = df1[chosen]
         y = df1['is_canceled']
         
         #the RandomForestClassifier was chosen after trials with other models proved that the RFC gave off the highest accuracy 
         rf_model = RandomForestClassifier(random_state =42)
         rf_model.fit(X,y)

         #define a function that predicts whether the customer with inserted booking details (in the sidebar) will cancel or not
         @st.cache(persist=True)
         def predict_cancel(hotel,lead_time,adr,total_of_special_requests,booking_changes,market_segment,deposit_type,assigned_room_type,customer_type,required_car_parking_spaces,previous_cancellations):
            input=np.array([[lead_time,adr,total_of_special_requests,booking_changes,market_segment,deposit_type,assigned_room_type,customer_type,required_car_parking_spaces,previous_cancellations]])
            prediction=rf_model.predict_proba(input)
            pred='{0:.{1}f}'.format(prediction[0][0], 2)
            return float(pred)

         #Creating the sidebar where booking details/parameters can be inserted to be used for prediction
         st.sidebar.text("Insert customer's booking details?")
         st.sidebar.markdown("#### Hotel")
         hotel = st.sidebar.selectbox("Choose the Branch",[0,1])
         st.sidebar.info("0 : Resort Hotel, 1 : City Hotel")
         st.sidebar.markdown("#### Lead Time")
         lead_time = st.sidebar.slider("Choose the Lead Time (Days)",0,400,step = 1)
         st.sidebar.markdown("#### Average Daily Rate (ADR)")
         adr = st.sidebar.slider("Choose the ADR", 0, 1000, step=1)
         st.sidebar.markdown("#### Total Special Requests")
         total_of_special_requests = st.sidebar.slider("Choose the number of special requests from guests", 0, 5, step=1)
         st.sidebar.markdown("#### Total Modifications")
         booking_changes = st.sidebar.slider("Choose the number of modifications made by guests", 0, 30, step=1)
         st.sidebar.markdown("#### Previous Cancellations By Guest")
         previous_cancellations = st.sidebar.slider("Choose the number of previous cancellations made by guests", 0, 30,step=1)
         st.sidebar.markdown("#### Market Segment")
         market_segment = st.sidebar.selectbox("Choose the Market Segment",[0,1,2,3,4,5,6,7])
         st.sidebar.info("0: Aviation, 1 : Complementary, 2: Corporate, 3: Direct, 4 : Groups, 5: Offline TA/TO, 6: Online TA,7: Undefined")
         st.sidebar.markdown("#### Deposit Type")
         deposit_type = st.sidebar.selectbox("Choose the Deposit Type",[0,1,2])
         st.sidebar.info("0 : No Deposit, 1 : Non Refund, 2 : Refundable")
         st.sidebar.markdown("#### Assigned Room Type")
         assigned_room_type = st.sidebar.selectbox("Choose the Assigned Room Type", [0,1,2,3,4,5,6,7,8,9,10,11])
         st.sidebar.info("0 : A, 1 : B, 2 : C, 3 : D, 4 : E, 5: F, 6 : G, 7 : H, 8 : I, 9 : K, 10 : L, 11 : P")
         st.sidebar.markdown("#### Customer Type")
         customer_type = st.sidebar.selectbox("Choose the Customer Type", [0,1,2,3])
         st.sidebar.info("0 : Contract, 1 : Group, 2 : Transient, 3 : Transient-Party")
         st.sidebar.markdown("#### Car Parking")
         required_car_parking_spaces = st.sidebar.selectbox("Choose the number of Car Parking Spaces", [0,1,2,3,4,5,6,7,8])

         #Give Prediciton when button is clicked
         if st.button("Predict If Customer will Cancel or Not"):
             output = predict_cancel(hotel,lead_time,adr,total_of_special_requests,booking_changes,market_segment,deposit_type,assigned_room_type,customer_type,required_car_parking_spaces,previous_cancellations)
             final_output = output * 100
             st.write('Probability of Guest Cancelling Reservation is {}% '.format(final_output))
             if final_output > 50.0:
                 st.error("This Customer's reservation is highly likely to be cancelled")
                 st.write("We recommend you contact the customer, understand if they have certain problems that you might able to address and tailor your offer accordingly. The direct contact proves premium customer service & attention. We also suggest you start enforcing Deposits with every booking.")
             else:
                 st.success("This Customer's reservation is not likely to be cancelled. You will welcome them soon!")

     else:
         #Show warning if dataset not uploaded yet
         st.warning("Please Upload a Dataset in the Data Upload Slot Above")
