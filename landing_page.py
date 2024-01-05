import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu

import yaml
from yaml.loader import SafeLoader
import time

from utils import initialise_agent
from langchain.callbacks import StreamlitCallbackHandler


from dotenv import load_dotenv
load_dotenv()


# st.set_page_config(layout='wide')

with open('./auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name,authentication_status,username = authenticator.login('Login', 'main')

if 'form_submit_button' not in st.session_state:
    st.session_state.form_submit_button = False

# if 'build_portfolio' not in st.session_state:
#     st.session_state.build_portfolio = False
# if 'exp_amt' not in st.session_state:
#     st.session_state.exp_amount = 0

# if 'investment_horizon' not in st.session_state:
#     st.session_state.investment_horizon = 0

# if 'conviction' not in st.session_state:
#     st.session_state.conviction = None

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar', key='unique_key')
    st.sidebar.title(f'Welcome *{st.session_state["name"]}*')

    def option_menu_1_callback(key):
        selection = st.session_state[key] 
        return
    def option_menu_2_callback(key):
        selection = st.session_state[key] 
        return
    selected = option_menu(
        menu_title = None,
        options=['Portfolio Overview','New Portfolio','Chatbot'],
        icons= ['bar-chart-line','plus-square','chat-left-dots'],
        default_index=0,
        orientation='horizontal',
        key='option_menu_1',
        on_change=option_menu_1_callback
    )

    if selected == 'Portfolio Overview':
        st.title(f'{selected}')
    if selected == 'New Portfolio':

        selected_2 = option_menu(
            menu_title = None,
            options=['Recommended Portfolio','Build Your Portfolio'],
            icons= ['robot','palette'],
            default_index=0,
            orientation='horizontal',
            key='option_menu_2',
            on_change=option_menu_2_callback
        )
        if selected_2 == 'Recommended Portfolio':

            def questionnaire_callback():
                st.session_state.form_submit_button = True
                # TODO
                # pass to LLM to curate portfolio

                with st.spinner('Curating Portfolio...'):
                    time.sleep(5)
                return

            # with st.form(key='my_form'):
            #     slider_input = st.slider('My slider', 0, 10, 5, key='my_slider')
            #     checkbox_input = st.checkbox('Yes or No', key='my_checkbox')
                # submit_button = st.form_submit_button(label='Submit', on_click=form_callback)
            if not st.session_state.form_submit_button:
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

            
    if selected=='Chatbot':
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