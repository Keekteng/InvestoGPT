import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu

import yaml
from yaml.loader import SafeLoader
import time

from utils import initialise_agent
from langchain.callbacks import StreamlitCallbackHandler

import pandas as pd

from dotenv import load_dotenv
load_dotenv()


# st.set_page_config(layout='wide')

@st.cache_data
def fetch_funds():
    return pd.read_csv('data/processed/20231212/fund_detail.csv')

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

if 'rec_form_submit_button' not in st.session_state:
    st.session_state.rec_form_submit_button = False

if 'diy_form_submit_button' not in st.session_state:
    st.session_state.diy_form_submit_button = False

if 'option_menu_1' not in st.session_state:
    st.session_state.option_menu_1 = 'Porfolio Overview'
if 'option_menu_2' not in st.session_state:
    st.session_state.option_menu_2 = 'Recommended Portfolio'

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar', key='unique_key')
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')

    def option_menu_1_callback(key='option_menu_1'):
        return 
    def option_menu_2_callback(key='option_menu_2'):
        return 
    
    option_menu(
        menu_title = None,
        options=['Portfolio Overview','New Portfolio','Chatbot'],
        icons= ['bar-chart-line','plus-square','chat-left-dots'],
        default_index=0,
        orientation='horizontal',
        key='option_menu_1',
        on_change=option_menu_1_callback(key='option_menu_1')
    )
    
    if st.session_state.option_menu_1 == 'Portfolio Overview':
        st.title(f'{st.session_state.option_menu_1}')
    if st.session_state.option_menu_1 == 'New Portfolio':

        option_menu(
            menu_title = None,
            options=['Recommended Portfolio','Build Your Portfolio'],
            icons= ['robot','palette'],
            default_index=0,
            orientation='horizontal',
            key='option_menu_2',
            on_change=option_menu_2_callback(key='option_menu_2')
        )
        if st.session_state.option_menu_2 == 'Recommended Portfolio':

            def questionnaire_callback():
                st.session_state.rec_form_submit_button = True
                # TODO
                # pass to LLM to curate portfolio

                with st.spinner('Curating Portfolio...'):
                    time.sleep(5)
                return

            # with st.form(key='my_form'):
            #     slider_input = st.slider('My slider', 0, 10, 5, key='my_slider')
            #     checkbox_input = st.checkbox('Yes or No', key='my_checkbox')
                # submit_button = st.form_submit_button(label='Submit', on_click=form_callback)
            if not st.session_state.rec_form_submit_button:
                with st.form(key="questionnaire"):

                    goal = st.selectbox("Purpose of Investment",
                                        options=['Retirement','Housing','Education','Travel','Emergency Fund','General Wealth'],
                                        key='goal')

                    investment_horizon = st.slider("What is your investment horizon(years)?",0,40, 
                                                value=[10,40],
                                                key='investment_horizon')

                    risk_tolerance = st.select_slider('Choose an option that best describes your risk tolerance.', 
                                            options=['Conservative','Conservative to Moderate','Moderate','Moderate to Aggressive','Aggressive'],
                                            key='risk_tolerance')

                    conviction = st.text_area("Describe an ethical or social cause that you feel strongly about and how do you wish to contribute to it.",key='conviction')

                    preference = st.text_area("Specify any regions you are interested in having exposure to.",key='preference')

                    submitted = st.form_submit_button("Submit",on_click=questionnaire_callback)
        else:
            st.title("Construct your personal portfolio!")

            def diy_form_callback():
                # Check for valid input before submitting form
                fund_list = [st.session_state.fund_1,st.session_state.fund_2,st.session_state.fund_3,st.session_state.fund_4,st.session_state.fund_5]
                allocation_list = [st.session_state.allocation_1,st.session_state.allocation_2,st.session_state.allocation_3,st.session_state.allocation_4,st.session_state.allocation_5]
                selected_fund_list = []

                for fund in fund_list:
                    # Check if user selected duplicate funds
                    if fund and fund in selected_fund_list:
                        st.error('You cannot choose duplicate funds')
                        return
                    else:
                        selected_fund_list.append(fund)
                    
                if st.session_state.fund_1 or st.session_state.fund_2 or st.session_state.fund_3 or st.session_state.fund_4 or st.session_state.fund_5:
                    total_allocation = 0
                    for fund,allocation in zip(fund_list,allocation_list):
                        # Check if selected fund has a valid % allocation
                        if fund and allocation==0:
                            st.error('A fund cannot have 0% allocation')
                            return
                        elif fund:
                            total_allocation+=allocation
                    # Check if total allocation adds up to 100%                            
                    if total_allocation!=100:
                        st.error('Total allocation needs to add up to 100%')
                        return 
                    # Check if investment amount is valid
                    if st.session_state.diy_investment_amount < 1000:
                        st.error('Minimum Investment Amount is $1000')
                        return
                else:
                    st.error("Please select at least 1 fund!")
                    return
                
                # SUCCESS
                st.session_state.diy_form_submit_button = True
                with st.spinner('Constructing Portfolio...'):
                    time.sleep(5)
                return

            if not st.session_state.diy_form_submit_button:
                with st.form(key="diy_form"):
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
    try:
        if authenticator.register_user('Register User', preauthorization=False):
            with open('./auth.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)
