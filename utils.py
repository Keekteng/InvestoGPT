from langchain.llms import OpenAI
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

def initialise_agent():
    db = SQLDatabase.from_uri(f"postgresql+psycopg2://{os.environ['user']}:{os.environ['db_password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['db_name']}",include_tables=['fund_detail','performance','region','sector'])
    llm = OpenAI(model='gpt-3.5-turbo-instruct',openai_api_key=os.environ['openai_api_key'],temperature=0,streaming=True)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['openai_api_key'])


    with open('langchain/pronouns.json','r')as f:
        texts = json.load(f)['pronouns']

    vector_db = FAISS.from_texts(texts, embeddings)
    retriever = vector_db.as_retriever()

    tool_description = """This tool will help you understand the which entity the user is referring to before querying for proper nouns like type of sectors (Health Care), type of regions (North America),symbol of funds (VWEAX) and type of fund categories (Large Growth).
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
                    input_key='input',memory_key="history", return_messages=True,k=1
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
    