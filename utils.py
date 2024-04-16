from langchain.llms import OpenAI as LangChainOpenAI
from openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.utilities import SQLDatabase
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents.agent_toolkits import create_retriever_tool
import os
import json
import time
import re
import ast

import psycopg2
import streamlit as st
import plotly.graph_objects as go

import pandas as pd
import datetime
from dateutil import tz
from dateutil.relativedelta import relativedelta
from datetime import timedelta,date
import plotly.express as px
import plotly.io as pio
import numpy as np
from numerize import numerize

pio.templates[pio.templates.default].layout.colorway = ["#e8e8e2","#ea6060","#dc9566","#22b4a4","#e99b26"]

# Styling Page 
tab_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
font-size: 24px;
}
</style>
"""
expander_css = """
<style>
div[data-testid="stExpander"] details summary p{
    font-size: 1.5rem;
}}
"""
reduced_margin_css ="""<style>
        .reduce-margin {
            margin-bottom: -50px;
        }
        </style>
        """

file_id = "file-tit011QW2gdYtMYBICMhfLKd"
assistant_id = "asst_RnhT52BHdoOaFpmBq7crVKxz"

# Global Variable
today_date = (datetime.datetime.now())

def parse_json_markdown(json_string: str) -> list:
    # Try to find JSON string within first and last triple backticks
    match = re.search(r"""```       # match first occuring triple backticks
                          (?:json)? # zero or one match of string json in non-capturing group
                          (.*)```   # greedy match to last triple backticks""", json_string, flags=re.DOTALL|re.VERBOSE)

    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
    else:
        # If match found, use the content within the backticks
        json_str = match.group(1)

    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()

    # Parse the JSON string into a list of Python dictionary while allowing control characters by setting strict to False
    parsed = json.loads(json_str, strict=False)

    return parsed

def run_openai_assistant(user_input):
    json_response = None

    client = OpenAI(api_key=os.environ['openai_api_key'])

    # Create new thread to input user's input
    thread = client.beta.threads.create(
        messages = [
            {'role':'user',
            'content':user_input
            }
        ]
    )
    run = client.beta.threads.runs.create(
        thread_id = thread.id,
        assistant_id = assistant_id,
    )

    run_status = client.beta.threads.runs.retrieve(
                        thread_id = thread.id,
                        run_id = run.id
                )

    while run_status.status != 'completed':
        #time out for 5 seconds before checking status again
        time.sleep(5)
        run_status = client.beta.threads.runs.retrieve(
                            thread_id = thread.id,
                            run_id = run.id
                        )
        if run_status.status in ['failed','cancelled','expired']:
            # Delete current thread
            client.beta.threads.delete(thread_id=thread.id)
            return False
        
    response = client.beta.threads.messages.list(thread_id=thread.id).data[0].content[0].text.value

    try:
        json_response = parse_json_markdown(response)
    except Exception as e:
        print('Error in OpenAI Assistant Response')
        print(e)
        print(response)
        time.sleep(5)
    finally:
        # Delete Thread after getting response from OpenAI Assistant
        client.beta.threads.delete(thread_id=thread.id)

    return json_response

def initialise_agent():
    db = SQLDatabase.from_uri(f"postgresql+psycopg2://{os.environ['user']}:{os.environ['db_password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['db_name']}",include_tables=['fund_detail','performance','region','sector'])
    llm = LangChainOpenAI(model='gpt-3.5-turbo-instruct',openai_api_key=os.environ['openai_api_key'],temperature=0,streaming=True)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['openai_api_key'])

    with open('langchain/pronouns.json','r')as f:
        texts = json.load(f)['pronouns']

    vector_db = FAISS.from_texts(texts, embeddings)
    retriever = vector_db.as_retriever()

    tool_description = """This tool will help you understand the which entity the user is referring to before querying 
    for proper nouns like type of sectors (Health Care), type of regions (North America),symbol of funds (VWEAX) and type of fund categories (Large Growth).
    Use it only after you have used tried using the sql_get_similar_examples tool.
    """

    entity_search_retriever_tool = create_retriever_tool(
        retriever,
        name="entity_search",
        description=tool_description,
    )

    with open('langchain/few_shot_example_query.json','r') as f:
        few_shots = json.load(f)

    few_shot_docs = [
        Document(page_content=question, metadata={"sql_query": few_shots[question]})
        for question in few_shots.keys()
    ]
    vector_db = FAISS.from_documents(few_shot_docs, embeddings)
    retriever = vector_db.as_retriever()

    tool_description = """This tool will help you understand similar examples to adapt them to the user question.
    Input to this tool should be the user question.
    Always use this tool first.
    """

    few_shot_retriever_tool = create_retriever_tool(
        retriever, name="sql_get_similar_examples", description=tool_description
    )

    custom_tool_list = [few_shot_retriever_tool,entity_search_retriever_tool]



    prefix_template = '''You are an agent designed to interact with a PostgreSQL database.
    Given an input question, create a syntactically correct PostgresSQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".

    You have access to tools for interacting with the database as well as some extra tools to help you construct accurate queries.
    You will ALWAYS FIRST use the tool sql_get_similar_examples to get the similar examples. If the examples are enough to construct the query, you can build it. Otherwise, you will then look at the tables in the database to see what you can query. Then you should use entity_search tool to check that you are querying the right entities before querying the schema of the most relevant tables.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "No such funds." as the answer.

    Only use the information returned by the below tools to construct your final answer.

    '''

    format_instructions_template = """Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}], always look first in sql_get_similar_examples
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: The final answer to the original input question. If the last observation leading up to the final answer does not return any row of result from the database, you should return "No such funds." as your final answer. Otherwise include the symbol of the fund(s) in your final answer and format it in a concise and easy to understand manner with as least words as possible. 
    """

    suffix_template = """Begin!

    Chat History: {history}
    Question: {input}

    If the last observation leading up to the final answer does not return any row of result from the database, you should return "No such funds." as your final answer.
    {agent_scratchpad}"""  



    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={
                "memory": ConversationBufferWindowMemory(
                    input_key='input',memory_key="history", return_messages=True,k=5
                ),
                'handle_parsing_errors':"Check your output and make sure it conforms!"
            },
        input_variables = ['input','agent_scratchpad','history'],
        prefix=prefix_template,
        format_instructions=format_instructions_template,
        extra_tools=custom_tool_list,
        suffix=suffix_template,
        top_k=5,
        handle_parsing_errors=True,
    )

    return agent_executor

def validate_diy_portfolio():
    # Check for valid input before submitting form
    fund_list = [st.session_state.fund_1,st.session_state.fund_2,st.session_state.fund_3,st.session_state.fund_4,st.session_state.fund_5]
    allocation_list = [st.session_state.allocation_1,st.session_state.allocation_2,st.session_state.allocation_3,st.session_state.allocation_4,st.session_state.allocation_5]
    selected_fund_list = []

    for fund in fund_list:
        # Check if user selected duplicate funds
        if fund and fund in selected_fund_list:
            st.error('You cannot choose duplicate funds')
            return False
        else:
            selected_fund_list.append(fund)
        
    if st.session_state.fund_1 or st.session_state.fund_2 or st.session_state.fund_3 or st.session_state.fund_4 or st.session_state.fund_5:
        total_allocation = 0
        for fund,allocation in zip(fund_list,allocation_list):
            # Check if selected fund has a valid % allocation
            if fund and allocation==0:
                st.error('A fund cannot have 0% allocation')
                return False
            elif fund:
                total_allocation+=allocation
        # Check if total allocation adds up to 100%                            
        if total_allocation!=100:
            st.error('Total allocation needs to add up to 100%')
            return False
        # Check if investment amount is valid
        if st.session_state.diy_investment_amount < 1000:
            st.error('Minimum Investment Amount is $1000')
            return False
    else:
        st.error("Please select at least 1 fund!")
        return False
    
    return True

@st.cache_data
def get_prices(list_of_symbol,portfolio_created_date):
    try:

        processed_dir_path = 'data/processed'

        directories = [d for d in os.listdir(processed_dir_path) if os.path.isdir(os.path.join(processed_dir_path, d))]
        latest_date = max(datetime.datetime.strptime(d, '%Y%m%d') for d in directories)

        # Format the datetime object back to a string in the original format
        latest_date_str = latest_date.strftime('%Y%m%d')
        latest_folder_path = os.path.join(processed_dir_path, latest_date_str)
        price_df = pd.read_csv(f'{latest_folder_path}/price.csv')

        # Return the latest price for a list of symbols
        price_df['date'] = price_df['date'].astype('datetime64[ns]')
        price_df = price_df[(price_df['symbol'].isin(list_of_symbol))&(price_df['date']>=portfolio_created_date)]

        # Step 1: Group by the date column
        grouped = price_df.groupby('date')

        # Step 2: Filter away dates whereby we do not have price for all purchased asset
        filtered_grouped = grouped.filter(lambda x: len(x) ==len(list_of_symbol))

        return filtered_grouped
    
    except Exception as e:
        print(e)

@st.cache_data
def get_urls():
    with open('data/raw/20240223/urls/funds_url.json','r') as f:
        return json.load(f)
        
def get_portfolio():
    try:
        conn = psycopg2.connect(
            host=os.environ['host'],
            port=os.environ['port'],
            database=os.environ['db_name'],
            user=os.environ['user'],
            password=os.environ['db_password']
        )

        # Use a parameterized query to prevent SQL injection
        query = """
        SELECT *
        FROM public.portfolio AS portfolio
        WHERE portfolio.username = %s;
        """

        # Execute the query with the username as a parameter
        with conn.cursor() as cursor:
            cursor.execute(query, (st.session_state['username'],))
            rows = cursor.fetchall()

        # Convert the result to a pandas DataFrame
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
        df['date'] = df['date'].astype('datetime64[ns]')
        return df

    except psycopg2.OperationalError as ex:
        if 'Connection refused' not in str(ex):
            print(ex)

def create_diy_portfolio():
    user_portfolio_df = get_portfolio()
    num_porfolio = user_portfolio_df['portfolio_id'].nunique()
    new_portfolio_id = 1
    if today_date.weekday()==5:
        # cant place an order on saturday --> place it on the following monday
        place_order_date = (today_date + timedelta(days=2)).strftime('%Y%m%d')
    elif today_date.weekday()==6:
        # Cant place order on sunday --> place it on following monday
        place_order_date = today_date + timedelta(days=1).strftime('%Y%m%d')
    else:
        place_order_date = today_date.strftime('%Y%m%d')

    if num_porfolio>0:
        new_portfolio_id = num_porfolio+1

    try:
        conn = psycopg2.connect(
            host = os.environ['host'],
            port = os.environ['port'],
            database = os.environ['db_name'],
            user = os.environ['user'],
            password = os.environ['db_password']
        )

        cursor = conn.cursor()
        
        if st.session_state.fund_1:
            amt_allocated = st.session_state['diy_investment_amount']*(st.session_state['allocation_1']/100)
            cursor.execute(
            f'''
            INSERT INTO public.portfolio
            (date,portfolio_id,username,symbol,buy_price,units,amount_allocated)
            VALUES
            ('{place_order_date}',{new_portfolio_id},'{st.session_state['username']}','{st.session_state['fund_1']}',NULL,NULL,{amt_allocated});
            '''
            )
        
        if st.session_state.fund_2:
            amt_allocated = st.session_state['diy_investment_amount']*(st.session_state['allocation_2']/100)
            cursor.execute(
            f'''
            INSERT INTO public.portfolio
            (date,portfolio_id,username,symbol,buy_price,units,amount_allocated)
            VALUES
            ('{place_order_date}',{new_portfolio_id},'{st.session_state['username']}','{st.session_state['fund_2']}',NULL,NULL,{amt_allocated});
            '''
            )

        if st.session_state.fund_3:
            amt_allocated = st.session_state['diy_investment_amount']*(st.session_state['allocation_3']/100)
            cursor.execute(
            f'''
            INSERT INTO public.portfolio
            (date,portfolio_id,username,symbol,buy_price,units,amount_allocated)
            VALUES
            ('{place_order_date}',{new_portfolio_id},'{st.session_state['username']}','{st.session_state['fund_3']}',NULL,NULL,{amt_allocated});
            '''
            )

        if st.session_state.fund_4:
            amt_allocated = st.session_state['diy_investment_amount']*(st.session_state['allocation_4']/100)
            cursor.execute(
            f'''
            INSERT INTO public.portfolio
            (date,portfolio_id,username,symbol,buy_price,units,amount_allocated)
            VALUES
            ('{place_order_date}',{new_portfolio_id},'{st.session_state['username']}','{st.session_state['fund_4']}',NULL,NULL,{amt_allocated});
            '''
            )

        if st.session_state.fund_5:
            amt_allocated = st.session_state['diy_investment_amount']*(st.session_state['allocation_5']/100)
            cursor.execute(
            f'''
            INSERT INTO public.portfolio
            (date,portfolio_id,username,symbol,buy_price,units,amount_allocated)
            VALUES
            ('{place_order_date}',{new_portfolio_id},'{st.session_state['username']}','{st.session_state['fund_5']}',NULL,NULL,{amt_allocated});
            '''
            )

        conn.commit()
        
    except psycopg2.OperationalError as ex:
            if 'Connection refused' not in str(ex):
                print(ex)

def create_rec_portfolio():
    user_portfolio_df = get_portfolio()
    num_porfolio = user_portfolio_df['portfolio_id'].nunique()
    new_portfolio_id = 1
    if today_date.weekday()==5:
        # cant place an order on saturday --> place it on the following monday
        place_order_date = (today_date + timedelta(days=2)).strftime('%Y%m%d')
    elif today_date.weekday()==6:
        # Cant place order on sunday --> place it on following monday
        place_order_date = today_date + timedelta(days=1).strftime('%Y%m%d')
    else:
        place_order_date = today_date.strftime('%Y%m%d')

    if num_porfolio>0:
        new_portfolio_id = num_porfolio+1

    try:
        conn = psycopg2.connect(
            host = os.environ['host'],
            port = os.environ['port'],
            database = os.environ['db_name'],
            user = os.environ['user'],
            password = os.environ['db_password']
        )

        cursor = conn.cursor()
        
        for fund in st.session_state.recommended_portfolio:
            amt_allocated = st.session_state.rec_investment_amount*(float(fund['% Allocation'].rstrip('%'))/100)
            cursor.execute(
            f'''
            INSERT INTO public.portfolio
            (date,portfolio_id,username,symbol,buy_price,units,amount_allocated)
            VALUES
            ('{place_order_date}',{new_portfolio_id},'{st.session_state['username']}','{fund['Symbol']}',NULL,NULL,{amt_allocated});
            '''
            )

        conn.commit()
        
    except psycopg2.OperationalError as ex:
            if 'Connection refused' not in str(ex):
                print(ex)

def plot_individual_fund_returns(prices_df,selected_portfolio_df):
    # Handle the case where there is no price data yet for funds bought
    if len(prices_df)==0:
        portfolio_allocation_df = selected_portfolio_df
        portfolio_allocation_df['investment_amt'] = portfolio_allocation_df['amount_allocated']
    else:
        combined_df = pd.merge(left=prices_df,right=selected_portfolio_df[['symbol','buy_price','units','amount_allocated']],on='symbol')
        combined_df['investment_amt'] = combined_df['units']*combined_df['price']
        portfolio_allocation_df = combined_df

    # Plot percentage return for each funds
    symbols = portfolio_allocation_df['symbol'].unique()
    portfolio_allocation_df['perc_return'] = ((portfolio_allocation_df['investment_amt']-portfolio_allocation_df['amount_allocated'])/portfolio_allocation_df['amount_allocated'])*100

    fig = go.Figure()

    for symbol in symbols:
        symbol_df = portfolio_allocation_df[portfolio_allocation_df['symbol'] == symbol]
        fig.add_trace(go.Scatter(x=symbol_df['date'], y=symbol_df['perc_return'], mode='lines', name=symbol))

    fig.update_layout(title='Returns by Funds',
                    xaxis_title='Date',
                    yaxis_title='Percentage Return',
                    legend_title='Symbol',height=500,width=750,template='gridon')

    # Display the figure in Streamlit
    st.subheader('Simple Returns')
    st.plotly_chart(fig)
    return

def plot_geographical_allocation(df):
    fig = px.pie(df,names='region',values='investment_amt')
    fig.update_traces(textposition='inside',textinfo='label+percent')
    st.plotly_chart(fig,use_container_width=True)
    return

def plot_sector_allocation(df):
    fig = px.pie(df,names='sector',values='investment_amt')
    fig.update_traces(textposition='inside',textinfo='label+percent')
    st.plotly_chart(fig,use_container_width=True)
    return

def plot_asset_allocation(df,path,values):
    fig = px.sunburst(df,path=path,values=values)
    fig.update_traces(textinfo='label+percent entry')
    st.plotly_chart(fig,use_container_width=True)
    return

def plot_performance_tab(prices_df,selected_portfolio_df):
    # Handle the case where there is no price data yet for funds bought
    if len(prices_df)==0:
        daily_portfolio_df = selected_portfolio_df.groupby('date').agg({'amount_allocated':'sum'}).reset_index()
        if len(daily_portfolio_df==1):
            daily_portfolio_df = pd.concat([daily_portfolio_df,pd.DataFrame({'date':[today_date+timedelta(days=1)],'amount_allocated':[selected_portfolio_df['amount_allocated'].sum()]})])
            daily_portfolio_df['date'] = pd.to_datetime(daily_portfolio_df['date'])
        daily_portfolio_df['investment_amt'] = daily_portfolio_df['amount_allocated']
    else:
        combined_df = pd.merge(left=prices_df,right=selected_portfolio_df[['symbol','buy_price','units','amount_allocated']],on='symbol')
        combined_df['investment_amt'] = combined_df['units']*combined_df['price']
        daily_portfolio_df = combined_df.groupby('date')[['amount_allocated','investment_amt']].sum().reset_index()
    daily_portfolio_df.rename(columns={'amount_allocated':'Net Investment','investment_amt':'Investment Value'},inplace=True)

    # Plot Returns
    st.subheader('Returns')
    st.text("")
    st.text("")
    col1,col2,col3,col4,col5,col6,col7 = st.columns([1,20,20,20,20,20,20],gap='large')

    first_day_of_the_year = datetime.datetime(date.today().year,1,1)
    earliest_investment_date = daily_portfolio_df['date'].min()
    three_month_ago = pd.to_datetime(date.today() + relativedelta(months=-3))
    six_month_ago = pd.to_datetime(date.today() + relativedelta(months=-6))
    one_year_ago = pd.to_datetime(date.today() + relativedelta(months=-12))
    three_years_ago = pd.to_datetime(date.today() + relativedelta(months=-36))

    daily_portfolio_df = daily_portfolio_df.sort_values(by='date',ascending=True)    
    if len(daily_portfolio_df)!=0:
        # Calculting Absolute Returns
        if earliest_investment_date<=first_day_of_the_year:
            ytd_return = round(daily_portfolio_df[daily_portfolio_df['date']>=first_day_of_the_year]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=first_day_of_the_year]['Investment Value'].iloc[0],2)
            ytd_perc_return = (daily_portfolio_df[daily_portfolio_df['date']>=first_day_of_the_year]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=first_day_of_the_year]['Investment Value'].iloc[0])/daily_portfolio_df[daily_portfolio_df['date']>=first_day_of_the_year]['Investment Value'].iloc[0]
        else:
            ytd_return = None
            ytd_perc_return = None

        if earliest_investment_date<=three_month_ago:
            three_month_return = round(daily_portfolio_df[daily_portfolio_df['date']>=three_month_ago]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=three_month_ago]['Investment Value'].iloc[0],2)
            three_month_perc_return = (daily_portfolio_df[daily_portfolio_df['date']>=three_month_ago]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=three_month_ago]['Investment Value'].iloc[0])/daily_portfolio_df[daily_portfolio_df['date']>=three_month_ago]['Investment Value'].iloc[0]
        else:
            three_month_return = None
            three_month_perc_return = None

        if earliest_investment_date<=six_month_ago:
            six_month_return = round(daily_portfolio_df[daily_portfolio_df['date']>=six_month_ago]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=six_month_ago]['Investment Value'].iloc[0],2)
            six_month_perc_return = (daily_portfolio_df[daily_portfolio_df['date']>=six_month_ago]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=six_month_ago]['Investment Value'].iloc[0])/daily_portfolio_df[daily_portfolio_df['date']>=six_month_ago]['Investment Value'].iloc[0]
        else:
            six_month_return = None
            six_month_perc_return = None

        if earliest_investment_date<=one_year_ago:
            one_year_return = round(daily_portfolio_df[daily_portfolio_df['date']>=one_year_ago]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=one_year_ago]['Investment Value'].iloc[0],2)
            one_year_perc_return = (daily_portfolio_df[daily_portfolio_df['date']>=one_year_ago]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=one_year_ago]['Investment Value'].iloc[0])/daily_portfolio_df[daily_portfolio_df['date']>=one_year_ago]['Investment Value'].iloc[0]
        else:
            one_year_return = None
            one_year_perc_return = None

        if earliest_investment_date<=three_years_ago:
            three_year_return = round(daily_portfolio_df[daily_portfolio_df['date']>=three_years_ago]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=three_years_ago]['Investment Value'].iloc[0],2)
            three_year_perc_return = (daily_portfolio_df[daily_portfolio_df['date']>=three_years_ago]['Investment Value'].iloc[-1] - daily_portfolio_df[daily_portfolio_df['date']>=three_years_ago]['Investment Value'].iloc[0])/daily_portfolio_df[daily_portfolio_df['date']>=three_years_ago]['Investment Value'].iloc[0]
        else:
            three_year_return = None
            three_year_perc_return = None

        all_time_return = round(daily_portfolio_df['Investment Value'].iloc[-1]-daily_portfolio_df['Investment Value'].iloc[0],2)
        all_time_perc_return = (daily_portfolio_df['Investment Value'].iloc[-1] - daily_portfolio_df['Investment Value'].iloc[0])/daily_portfolio_df['Investment Value'].iloc[0]
    else:
        ytd_return = None
        three_month_return = None
        six_month_return = None
        one_year_return = None
        three_year_return = None
        all_time_return = None

        ytd_perc_return = None
        three_month_perc_return = None
        six_month_perc_return = None
        one_year_perc_return = None
        three_year_perc_return = None
        all_time_perc_return= None
    
    st.markdown(
        """
    <style>
    [data-testid="stMetricDelta"] {
        font-size: 15px;
    }
    </style>

    <style>
    [data-testid="stMetricValue"] {
        font-size: 20px;
    }
    </style>

    """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
    <style>
    div[data-testid="column"] {
    transform: scale(1.5);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    col2.metric(
            label='YTD',
            value = '' if ytd_return is None else (f'+{numerize.numerize(ytd_return)}' if ytd_return>=0 else f'{numerize.numerize(ytd_return)}'),
            delta=f"{round(ytd_perc_return*100,2)}%" if ytd_perc_return is not None else ''
        )

    col3.metric(
        label='3 Months',
        value = '' if three_month_return is None else (f'+{numerize.numerize(three_month_return)}' if three_month_return>=0 else f'{numerize.numerize(three_month_return)}'),
        delta=f"{round(three_month_perc_return*100,2)}%" if three_month_perc_return is not None else ''
    )

    col4.metric(
        label='6 Months',
        value = '' if six_month_return is None else (f'+{numerize.numerize(six_month_return)}' if six_month_return>=0 else f'{numerize.numerize(six_month_return)}'),
        delta=f"{round(six_month_perc_return*100,2)}%" if six_month_perc_return is not None else ''
    )

    col5.metric(
        label='1 Year',
        value = '' if one_year_return is None else (f'+{numerize.numerize(one_year_return)}' if one_year_return>=0 else f'{numerize.numerize(one_year_return)}'),
        delta=f"{round(one_year_perc_return*100,2)}%" if one_year_perc_return is not None else ''
    )

    col6.metric(
        label='3 Years',
        value = '' if three_year_return is None else (f'+{numerize.numerize(three_year_return)}' if three_year_return>=0 else f'{numerize.numerize(three_year_return)}'),
        delta=f"{round(three_year_perc_return*100,2)}%" if three_year_perc_return is not None else ''
    )
    
    col7.metric(
        label='All Time',
        value='' if all_time_return is None else (f'+{numerize.numerize(all_time_return)}' if all_time_return>=0 else f'{numerize.numerize(all_time_return)}'),
        delta=f'{round(all_time_perc_return*100,2)}%' if all_time_perc_return is not None else ''
    )

    # Plot Portfolio Graph
    st.subheader('Portfolio Value Over Time 📈')
    fig = px.line(daily_portfolio_df,x='date',y=['Net Investment','Investment Value'],labels={'date':'Date','value':'Value'},height=500,width=750,template='gridon')
    st.plotly_chart(fig)

    plot_individual_fund_returns(prices_df,selected_portfolio_df)

    return

def plot_portfolio_tab(prices_df,selected_portfolio_df,fund_detail_df):

    # Handle the case where there is no price data yet for funds bought
    if len(prices_df)==0:
        portfolio_allocation_df = selected_portfolio_df
        portfolio_allocation_df['investment_amt'] = portfolio_allocation_df['amount_allocated']
    else:
        combined_df = pd.merge(left=prices_df,right=selected_portfolio_df[['symbol','buy_price','units','amount_allocated']],on='symbol')
        combined_df['investment_amt'] = combined_df['units']*combined_df['price']
        portfolio_allocation_df = combined_df
    portfolio_allocation_df = pd.merge(left=portfolio_allocation_df,right=fund_detail_df[['symbol','asset_class','region','sector']],on='symbol')
    latest_portfolio_allocation_df = portfolio_allocation_df[portfolio_allocation_df['date']==portfolio_allocation_df['date'].max()]
    plot_asset_allocation(latest_portfolio_allocation_df,path=['asset_class','symbol'],values='investment_amt')
    latest_portfolio_allocation_df['sector'] = latest_portfolio_allocation_df['sector'].apply(lambda d: d if isinstance(d, str) else "['Others']")
    
    region_asset_dict = {}
    sector_asset_dict = {}


    for inv_amt,region_list,sector_list in zip(latest_portfolio_allocation_df['investment_amt'].tolist(),latest_portfolio_allocation_df['region'].tolist(),latest_portfolio_allocation_df['sector']):
        for region in ast.literal_eval(region_list):
            if region in region_asset_dict:
                region_asset_dict[region]+=inv_amt
            else:
                region_asset_dict[region] = inv_amt
        for sector in ast.literal_eval(sector_list):
            if sector in sector_asset_dict:
                sector_asset_dict[sector] +=inv_amt
            else:
                sector_asset_dict[sector] = inv_amt

    region_allocation_df = pd.DataFrame(list(region_asset_dict.items()),columns=['region','investment_amt'])
    sector_allocation_df = pd.DataFrame(list(sector_asset_dict.items()),columns=['sector','investment_amt'])
    st.subheader('Geographical Allocation')
    plot_geographical_allocation(region_allocation_df)
    st.subheader('Sector Allocation')
    plot_sector_allocation(sector_allocation_df)

def plot_underlying_fund_tab(selected_portfolio_df):
    underlying_fund_df = pd.read_csv('langchain/fund_details.csv').drop(columns=['Unnamed: 0'])
    performance_df = pd.read_csv('data/processed/20240223/perf.csv').drop(columns=['Unnamed: 0'])
    funds_url_dict = get_urls()
    combined_df = pd.merge(right=underlying_fund_df,left=selected_portfolio_df[['symbol']],on='symbol')
    combined_df = pd.merge(left=combined_df,right=performance_df,on='symbol')
    combined_df['sector'] = combined_df['sector'].apply(lambda d: d if isinstance(d, str) else "['Others']")
    combined_df['region'] = combined_df['region'].apply(lambda d: d if isinstance(d, str) else "['Others']")
    combined_df = combined_df.rename(columns={'fund_return_ytd':'YTD','average_annual_fund_return_for_1_year':'1 Yr','average_annual_fund_return_for_3_year':'3 Yrs','average_annual_fund_return_for_5_year':'5 Yrs','average_annual_fund_return_for_10_year':'10 Yrs'})

    st.markdown(expander_css, unsafe_allow_html=True)
    # subheader with reduced margin
    st.markdown(reduced_margin_css, unsafe_allow_html=True)
    for fund in combined_df['symbol'].tolist():
        with st.expander(f"{fund}"):
            product_summary = combined_df[combined_df['symbol']==fund]['product_summary'].values[0]
            region_exposure = ast.literal_eval(combined_df[combined_df['symbol']==fund]['region'].values[0])
            sector_exposure = ast.literal_eval(combined_df[combined_df['symbol']==fund]['sector'].values[0])
            risk_level = combined_df[combined_df['symbol']==fund]['risk_level'].values[0]
            if product_summary:
                st.markdown('<h3 class="reduce-margin">Summary 📝</h3>', unsafe_allow_html=True)
                st.markdown('---')
                st.write(product_summary)
            if region_exposure:
                st.markdown('<h3 class="reduce-margin">Region Allocation 🌎</h3>', unsafe_allow_html=True)
                st.markdown('---')
                st.write(', '.join(region_exposure))
            if isinstance(sector_exposure,list):
                if not pd.isna(sector_exposure).any():
                    st.markdown('<h3 class="reduce-margin">Sector Allocation 🏗</h3>', unsafe_allow_html=True)
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
            
            if len(combined_df)>0:
                st.markdown('<h3 class="reduce-margin">Past Returns 📊</h3>', unsafe_allow_html=True)
                st.markdown('---')
                st.write(combined_df[['symbol','YTD','1 Yr','3 Yrs','5 Yrs','10 Yrs']].set_index('symbol'))
            
            st.markdown('<h3 class="reduce-margin">External Links 🔗</h3>', unsafe_allow_html=True)
            st.markdown('---') 

            st.link_button(
                label='  :link: Vanguard  ', 
                url=funds_url_dict[fund], 
            )