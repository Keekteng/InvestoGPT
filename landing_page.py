import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
import plotly.express as px
import yaml
from yaml.loader import SafeLoader
import time
import ast
import numpy as np
import json

from utils import initialise_agent,create_diy_portfolio,validate_diy_portfolio,get_portfolio,get_prices,plot_performance_tab,run_openai_assistant,plot_asset_allocation\
                ,create_rec_portfolio,plot_portfolio_tab,plot_underlying_fund_tab,tab_css,expander_css,reduced_margin_css
from baseline_portfolio import *
from langchain.callbacks import StreamlitCallbackHandler

import pandas as pd

from dotenv import load_dotenv
load_dotenv()


# st.set_page_config(layout='wide')

@st.cache_data
def fetch_funds():
    return pd.read_csv('langchain/fund_details.csv')


# Custom function to safely convert string to list
def convert_to_list(value):
    if pd.isnull(value):
        return value # Return the original value if it's null
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value # Return the original value if conversion fails


fund_detail_df = fetch_funds()

with open('./auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']

)


name,authentication_status,username = authenticator.login('Login', 'main')

if 'created_portfolio' not in st.session_state:
    st.session_state.created_portfolio = False

if 'username' not in st.session_state:
    st.session_state.username = username

if 'rec_form_submit_button' not in st.session_state:
    st.session_state.rec_form_submit_button = False

if 'diy_form_submit_button' not in st.session_state:
    st.session_state.diy_form_submit_button = False

if 'option_menu_1' not in st.session_state:
    st.session_state.option_menu_1 = 'Porfolio Overview'
if 'option_menu_2' not in st.session_state:
    st.session_state.option_menu_2 = 'Recommended Portfolio'

if 'recommended_portfolio' not in st.session_state:
    st.session_state.recommended_portfolio = False

if 'count' not in st.session_state:
    st.session_state.count = 0

if 'pages' not in st.session_state:
    st.session_state.pages = [
        "recommended_portfolio_overview_page",
        "confirm_investment_page",
                    ]

if st.session_state["authentication_status"]:

    authenticator.logout('Logout', 'sidebar', key='unique_key')
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')
    
    option_menu(
        menu_title = None,
        options=['Portfolio Overview','New Portfolio','Chatbot'],
        icons= ['bar-chart-line','plus-square','chat-left-dots'],
        default_index=0,
        orientation='horizontal',
        key='option_menu_1',
    )
    
    if st.session_state.option_menu_1 == 'Portfolio Overview':

        user_portfolio_df = get_portfolio()

        selection_list = ['Portfolio '+ str(x) for x in user_portfolio_df.sort_values(by='portfolio_id')['portfolio_id'].unique()]
        selected_portfolio = st.selectbox('',
                            options=selection_list,
                            placeholder='Select a Portfolio',
                            index=None,
                            key='portfolio_select_box')
        portfolio_id = None
        if selected_portfolio:
            portfolio_id = int(selected_portfolio[-1])

        selected_portfolio_df = user_portfolio_df[user_portfolio_df['portfolio_id']==portfolio_id]
        tab_list = ['Performance','Portfolio Allocation','Underlying Funds']
        whitespace = 10
        tabs = st.tabs([s.center(whitespace,"\u2001") for s in tab_list])
        st.markdown(tab_css, unsafe_allow_html=True)

        prices_df = get_prices(selected_portfolio_df['symbol'].unique(),selected_portfolio_df['date'].min())
        with tabs[0]:
            plot_performance_tab(prices_df,selected_portfolio_df)

        with tabs[1]:
            st.subheader("Current Asset Allocation")
            plot_portfolio_tab(prices_df,selected_portfolio_df,fund_detail_df)

        with tabs[2]:
            plot_underlying_fund_tab(selected_portfolio_df)

    if st.session_state.option_menu_1 == 'New Portfolio':

        option_menu(
            menu_title = None,
            options=['Recommended Portfolio','Build Your Portfolio'],
            icons= ['robot','palette'],
            default_index=0,
            orientation='horizontal',
            key='option_menu_2',
        )
        # Recommended portfolio section
        if st.session_state.option_menu_2 == 'Recommended Portfolio':

            def questionnaire_callback():

                if st.session_state['investment_horizon']<=2 and st.session_state['risk_tolerance']!='Conservative':
                    st.warning('We recommend a more conservative approach. At your current risk appetite, you might be less likely to achieve your goal in your time horizon as you have less time to recover from potential losses.', icon="⚠️")
                elif st.session_state['investment_horizon']<=4 and st.session_state['risk_tolerance'] not in ['Conservative','Conservative to Moderate']:
                    st.warning('We recommend a more conservative approach. At your current risk appetite, you might be less likely to achieve your goal in your time horizon as you have less time to recover from potential losses.', icon="⚠️")
                elif st.session_state['investment_horizon']<=7 and st.session_state['risk_tolerance'] not in ['Conservative','Conservative to Moderate','Moderate']:
                    st.warning('We recommend a more conservative approach. At your current risk appetite, you might be less likely to achieve your goal in your time horizon as you have less time to recover from potential losses.', icon="⚠️")
                elif st.session_state['investment_horizon']<=9 and st.session_state['risk_tolerance'] not in ['Conservative','Conservative to Moderate','Moderate','Moderate to Aggressive']:
                    st.warning('We recommend a more conservative approach. At your current risk appetite, you might be less likely to achieve your goal in your time horizon as you have less time to recover from potential losses.', icon="⚠️")
                else:
                    user_input = None
                    baseline = None
                    baseline_json = None
                    # Retrieve Baseline Portfolio based on user's Investment Time Horizon & Risk Tolerance & Append to User's Investment Preference
                    if st.session_state['risk_tolerance']=='Conservative':
                        baseline = conservative_baseline
                        baseline_json = json_conservative_baseline
                    elif st.session_state['risk_tolerance'] == 'Conservative to Moderate':
                        baseline = conservative_moderate_baseline
                        baseline_json = json_conservative_moderate_baseline
                    elif st.session_state['risk_tolerance'] == 'Moderate':
                        baseline = moderate_baseline
                        baseline_json = json_moderate_baseline
                    elif st.session_state['risk_tolerance'] == 'Moderate to Aggressive':
                        baseline = moderate_aggressive_baseline
                        baseline_json = json_moderate_aggressive_baseline
                    elif st.session_state['risk_tolerance'] == 'Aggressive':
                        baseline = aggressive_baseline
                        baseline_json = json_aggressive_baseline

                    user_input = baseline + f"""User's Investment Preference\n\"\"\"{st.session_state['conviction']}\n{st.session_state['preference']}\"\"\" """
                    st.success('Your risk appetite is suitable for your time horizon!', icon="✅")
                    with st.spinner('Curating Portfolio...'):
                        if st.session_state['conviction'] =="" and st.session_state['preference']=='':
                            response = json.loads(baseline_json)
                            time.sleep(1.5)
                        else:
                            response = run_openai_assistant(user_input)

                        if response:
                            # Save recommended portfolio into session_state
                            st.session_state.recommended_portfolio = response
                            st.session_state.rec_form_submit_button = True

                        else:
                            st.error('Try again, request failed.')
                            time.sleep(2)

                return

            if not st.session_state.rec_form_submit_button:
                with st.form(key="questionnaire"):

                    investment_horizon = st.slider("What is your investment horizon(years)?",1,40, 
                                                key='investment_horizon'
                                                )

                    risk_tolerance = st.select_slider('Choose an option that best describes your risk tolerance.', 
                                            options=['Conservative','Conservative to Moderate','Moderate','Moderate to Aggressive','Aggressive'],
                                            key='risk_tolerance')

                    conviction = st.text_area("Describe an ethical or social cause that you feel strongly about and how do you wish to contribute to it.",key='conviction')

                    preference = st.text_area("Specify any regions you are interested in having exposure to.",key='preference')

                    submitted = st.form_submit_button("Submit",on_click=questionnaire_callback)
            
            else:

                def display_page():
                    if st.session_state.pages[st.session_state.count]=='recommended_portfolio_overview_page' and st.session_state.recommended_portfolio:
                        underlying_fund_df = pd.read_csv('langchain/fund_details.csv')
                        df = pd.DataFrame(st.session_state.recommended_portfolio)
                        df['% Allocation'] = df['% Allocation'].str.rstrip('%').astype(float)
                        underlying_fund_df['product_summary'] = underlying_fund_df['product_summary'].str.replace(r'Important note:.*','',regex=True)
                        underlying_fund_df['region'] = underlying_fund_df[underlying_fund_df['region'].notna()]['region'].apply(convert_to_list)
                        underlying_fund_df['sector'] = underlying_fund_df[underlying_fund_df['sector'].notna()]['sector'].apply(convert_to_list)


                        st.subheader('Target Asset Allocation')
                        plot_asset_allocation(df,path=['Asset Class','Symbol'],values='% Allocation')

                        st.subheader('Underlying Funds')
                        st.markdown(expander_css, unsafe_allow_html=True)
                        st.markdown(reduced_margin_css, unsafe_allow_html=True)
                        for fund in df['Symbol'].tolist():
                            with st.expander(f"{fund}"):
                                perc_allocation = df[df['Symbol']==fund]['% Allocation'].values[0]
                                product_summary = underlying_fund_df[underlying_fund_df['symbol']==fund]['product_summary'].values[0]
                                region_exposure = underlying_fund_df[underlying_fund_df['symbol']==fund]['region'].values[0]
                                sector_exposure = underlying_fund_df[underlying_fund_df['symbol']==fund]['sector'].values[0]
                                risk_level = underlying_fund_df[underlying_fund_df['symbol']==fund]['risk_level'].values[0]
                                if perc_allocation:
                                    st.markdown('<h3 class="reduce-margin">% Allocation</h3>', unsafe_allow_html=True)
                                    st.markdown('---')
                                    st.write(str(perc_allocation)+"%")
                                if product_summary:
                                    st.markdown('<h3 class="reduce-margin">Summary 📝</h3>', unsafe_allow_html=True)
                                    st.markdown('---')
                                    st.write(product_summary)
                                if region_exposure:
                                    st.markdown('<h3 class="reduce-margin">Regions 🌎</h3>', unsafe_allow_html=True)
                                    st.markdown('---')
                                    st.write(', '.join(region_exposure))
                                if isinstance(sector_exposure,list):
                                    if not pd.isna(sector_exposure).any():
                                        st.markdown('<h3 class="reduce-margin">Sectors 🏗</h3>', unsafe_allow_html=True)
                                        st.markdown('---')
                                        st.write(', '.join(sector_exposure))
                                else:
                                    if not pd.isna(sector_exposure):
                                        st.markdown('<h3 class="reduce-margin">Sectors 🏗</h3>', unsafe_allow_html=True)
                                        st.markdown('---')
                                        st.write(', '.join(sector_exposure))
                                if risk_level:
                                    st.markdown('<h3 class="reduce-margin">Risk Level 🚨</h3>', unsafe_allow_html=True)
                                    st.markdown('---')
                                    if risk_level==1:
                                        st.write('Conservative')
                                    elif risk_level==2:
                                        st.write('Conservative to Moderate')
                                    elif risk_level==3:
                                        st.write('Moderate')
                                    elif risk_level==4:
                                        st.write("Moderate to Aggressive")
                                    else:
                                        st.write('Aggressive')
                    else:
                        rec_investment_amount = st.number_input(label="Investment Amount",step=1.,format="%.2f",key='rec_investment_amount')

                    return

                def next_page():
                    st.session_state.count += 1

                def previous_page():
                    if st.session_state.count == 0:
                        st.session_state.rec_form_submit_button = False
                    else:
                        st.session_state.count -= 1

                def confirm_investment():
                    if st.session_state.rec_investment_amount<1000:
                        st.error("Minimum Investment Amount is $1000")
                    else:
                        create_rec_portfolio()
                        st.session_state.count = 0
                        st.session_state.rec_form_submit_button = False
                        st.success('Portfolio Ready! View under Portfolio Overview Section!', icon="✅")
                        time.sleep(3)

                display_page()

                col1, col2,col3,col4,col5,col6,col7,col8 = st.columns([2,1,1,1,1,1,2,2])

                with col1:
                    if st.button("⏮️ Previous", on_click=previous_page):
                        pass

                with col8:
                    if st.session_state.pages[st.session_state.count]=='recommended_portfolio_overview_page':
                        if st.button("Next ⏭️", on_click=next_page):
                            pass
                    elif st.button('Confirm ✅',on_click=confirm_investment):
                        pass

        # DIY Portfolio Section
        else:
            st.title("Construct your personal portfolio!")

            def diy_form_callback():
                if validate_diy_portfolio():
                    with st.spinner('Constructing Portfolio...'):
                        create_diy_portfolio()
                        st.session_state.created_portfolio = True
                        time.sleep(1.5)
                return

            # form for diy portfolio
            with st.form(key="diy_form",clear_on_submit=True):
                c1,c2 = st.columns([5,2])
                with c1:
                    fund_1 = st.selectbox("Fund 1",
                                        options=fund_detail_df['symbol'],
                                        placeholder='Choose a fund',
                                        index=None,
                                        key='fund_1')
                    
                    fund_2 = st.selectbox("Fund 2",
                                        options=fund_detail_df['symbol'],
                                        placeholder='Choose a fund',
                                        index=None,
                                        key='fund_2')   
                            
                    fund_3 = st.selectbox("Fund 3",
                                        options=fund_detail_df['symbol'],
                                        placeholder='Choose a fund',
                                        index=None,                                            
                                        key='fund_3')

                    fund_4 = st.selectbox("Fund 4",
                                        options=fund_detail_df['symbol'],
                                        placeholder='Choose a fund',
                                        index=None,
                                        key='fund_4')
                    
                    fund_5 = st.selectbox("Fund 5",
                                        options=fund_detail_df['symbol'],
                                        placeholder='Choose a fund',
                                        index=None,
                                        key='fund_5')                        
                with c2:
                    allocation_1 = st.number_input("% Allocation",0,100, 
                                                key='allocation_1')
                    
                    allocation_2 = st.number_input("% Allocation",0,100, 
                                                key='allocation_2')
                    
                    allocation_3 = st.number_input("% Allocation",0,100, 
                                                key='allocation_3')
                    
                    allocation_4 = st.number_input("% Allocation",0,100, 
                                                key='allocation_4')
                    
                    allocation_5 = st.number_input("% Allocation",0,100, 
                                                key='allocation_5')
                    
                diy_investment_amount = st.number_input(label="Investment Amount",step=1.,format="%.2f",key='diy_investment_amount')

                submitted = st.form_submit_button("Submit",on_click=diy_form_callback)
                if st.session_state.created_portfolio:
                    # Change back to false
                    st.success('Portfolio ready! View under Portfolio Overview.', icon="✅")
                    st.session_state.created_portfolio = False
                    time.sleep(3)

    # Chatbot     
    if st.session_state.option_menu_1=='Chatbot':
        st.header('Explore Our Funds: Engage with Our Chatbot Today!')
        if 'agent_executor' not in st.session_state:
            agent_executor = initialise_agent()
            st.session_state.agent_executor = agent_executor
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Sample Question: What are the top 5 equity funds in terms of year to date return?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = st.session_state.agent_executor.run(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

if not authentication_status:
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
    st.cache_data.clear()
    try:
        if authenticator.register_user('Register User', preauthorization=False):
            with open('./auth.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)
