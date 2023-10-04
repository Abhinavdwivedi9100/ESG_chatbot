import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import Error
import sqlalchemy as sa
import openai
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import openai, pinecone, os
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext, GPTVectorStoreIndex, set_global_service_context
from llama_index.vector_stores import PineconeVectorStore
from langchain.chat_models import ChatOpenAI


api_key = "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"
llm = OpenAI(temperature=0, openai_api_key=api_key)
pinecone_index = 'esg'
pinecone.init(api_key="55adc9d0-b99f-4037-bf17-976c1e0790c1", environment="us-east-1-aws")
openai.api_key = "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"

def execute_sql_query(query):
    db_params = {
        'dbname': 'llm_test',
        'user': 'fleet_dba',
        'password': 'R2K34lYlYrFghithAzYGeBsroseTA332o4x093-tLWwc8pzzCcasf',
        'host': 'fleet.c7bt89b55bcg.us-east-1.rds.amazonaws.com',
        'port': '5432'
    }
    connection = None
    try:
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        # Convert results to list of dictionaries
        results_as_dict = [dict(zip(column_names, row)) for row in results]
        return results_as_dict
    except Error as e:
        print(f"Error: {e}")
        raise
    finally:
        if connection:
            cursor.close()
            connection.close()

def get_all_tables():
    query = """
        SELECT DISTINCT table_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """
    try:
        results = execute_sql_query(query)
        tables = [row['table_name'] for row in results]  # Extract 'table_name' from each dictionary
        return tables
    except Exception as e:
        print(e)
        return None

def get_table_schema(table_name):
    query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = '{table_name}'
        ORDER BY ordinal_position;
    """
    try:
        results = execute_sql_query(query)
        schema = {
            "table_name": table_name,
            "columns": [{"name": column['column_name'], "data type": column['data_type']} for column in results]
        }
        return schema
    except Exception as e:
        print(e)
        return None
    
def save_schema_to_file(schema, filename):
    try:
        with open(filename, 'w') as file:
            file.write(str(schema))
        print(f"Schema saved to '{filename}' successfully.")
    except Exception as e:
        print(f"Error saving schema to '{filename}': {e}")

def generate_sql_query(question, table_name):
    sch = get_table_schema(table_name)
    prompt = f"Act as a postgresql specialist to Generate the complete postgresqlquery (keep all the character columns on left) using the following schema:\n\n{sch}\n\nFor the following question:\n{question}\nConsider all the column names in double qoutes.\nSQL Query:\n"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        api_key=api_key
    )
    
    sql_query = response.choices[0].text.strip()
    print (sql_query)
    return sql_query
