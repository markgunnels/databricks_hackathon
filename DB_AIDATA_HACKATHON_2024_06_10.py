# Databricks notebook source
!pip install dspy-ai

# COMMAND ----------


"""
Set table locations for claims processing data
"""

diagnosis = "healthverity_claims_sample_patient_dataset.hv_claims_sample.diagnosis"
enrollment = "healthverity_claims_sample_patient_dataset.hv_claims_sample.enrollment"
medical_claim = "healthverity_claims_sample_patient_dataset.hv_claims_sample.medical_claim"
pharmacy_claim = "healthverity_claims_sample_patient_dataset.hv_claims_sample.pharmacy_claim"
procedure = "healthverity_claims_sample_patient_dataset.hv_claims_sample.procedure"
provider = "healthverity_claims_sample_patient_dataset.hv_claims_sample.provider"

# COMMAND ----------


"""
Register tables as temporary views for ease of use
"""

from datetime import datetime

# register tables as temporary views
for _table in (diagnosis, enrollment, medical_claim, pharmacy_claim, procedure, provider):
  try:
    _name = _table.split(".")[-1]
    _df = spark.table(_table)
    _df.createOrReplaceTempView(_name)
    print(f"{datetime.now()} : Registered {_table} as {_name}")
  except Exception as err:
    raise ValueError(f"{datetime.now()} : Failed to register {_table} as {_name}, err : {err}")
else:
  print(f"{datetime.now()} : Finished registering tables. All good to proceed!")

# COMMAND ----------

# Create a dataframe for the Catalog table
catalog_table = spark.table("pharmacy_claim")

# Display the dataframe
display(catalog_table)

# COMMAND ----------

import requests

def fetch_drug_label_data(rxcui):
    """Fetches drug label data from the FDA API for a given RXCUI.

    Args:
        rxcui (str): The RXCUI (RxNorm Concept Unique Identifier) of the drug.

    Returns:
        dict: The JSON data returned by the FDA API, or None if an error occurs.
    """

    url = f'https://api.fda.gov/drug/label.json?search=openfda.rxcui.exact:{rxcui}&limit=1'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for error status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching drug label data: {e}")
        return None

# COMMAND ----------

def ndc_to_rxnorm(ndc:str) -> str:
    url = f'https://rxnav.nlm.nih.gov/REST/rxcui.json?idtype=NDC&id={ndc}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for error status codes
        return response.json()['idGroup']['rxnormId'][0]
    except Exception as e:
        print(f"Error fetching drug label data: {e}")
        return e
    
ndc_to_rxnorm('65862018730')

# COMMAND ----------

import logging
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


logging.basicConfig(filename='udf.log', level=logging.INFO)

# COMMAND ----------

def ndcs_to_rxnorms(ndcs: list[str]) -> list[str]:
    rxnorms = []
    logging.info(ndcs)
    print(ndcs)
    for ndc in ndcs:
        print(ndc)
        rxnorms.append(ndc_to_rxnorm(ndc))
    
    return rxnorms

ndcs_to_rxnorms(["65862018730","67877032101","72578000805","65162046450"])

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

ndcs_to_rxnorms_udf = udf(ndcs_to_rxnorms, ArrayType(StringType()))

# COMMAND ----------

from pyspark.sql.functions import collect_list

grouped_df = catalog_table.groupBy("patient_id").agg(collect_list("NDC").alias("NDCs"))
display(grouped_df)

# COMMAND ----------

import pandas as pd

pandas_df = grouped_df.toPandas()

# COMMAND ----------

import pandas as pd
t_df = pandas_df[0:5]
# Apply the function to create a new column
t_df['rxcuis'] = t_df['NDCs'].apply(ndcs_to_rxnorms)

# Display the updated DataFrame
t_df

# COMMAND ----------

def fetch_drug_label_data(rxcui):
    url = f'https://api.fda.gov/drug/label.json?search=openfda.rxcui.exact:{rxcui}&limit=1'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for error status codes
        return response.json()['results'][0]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching drug label data: {e}")
        return None
    
dl = fetch_drug_label_data('855332')

dl.keys()
# drug_interactions = dl['results'][0]['drug_interactions'][0]

# COMMAND ----------

dl['openfda']

# COMMAND ----------

def fetch_drug_labels_information(rxcuis):
    drug_labels = []

    for rxcui in rxcuis:
        try:
            drug_labels.append(fetch_drug_label_data(rxcui))
        except:
            continue

    return drug_labels


# COMMAND ----------

t_df['drug_labels'] = t_df['rxcuis'].apply(fetch_drug_labels_information)

# COMMAND ----------

t_df

# COMMAND ----------

!pip install typing_extensions==4.7.1 --upgrade

# COMMAND ----------

import dspy
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
print(len(DATABRICKS_TOKEN))

# COMMAND ----------

prompt = 'SAY HELLO'
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

lm = dspy.Databricks(model='databricks-meta-llama-3-70b-instruct', 
                     api_key = DATABRICKS_TOKEN, 
                     api_base = 'https://dbc-0014f6a7-c498.cloud.databricks.com/serving-endpoints')

lm(prompt)

# COMMAND ----------

from openai import OpenAI
import os

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
# Alternatively in a Databricks notebook you can use this:
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://dbc-0014f6a7-c498.cloud.databricks.com/serving-endpoints"
)

chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": "Tell me about Large Language Models"
  }
  ],
  model="databricks-llama-2-70b-chat",
  max_tokens=256
)

print(chat_completion.choices[0].message.content)

# COMMAND ----------

# MAGIC %pip install databricks-genai-inference
# MAGIC

# COMMAND ----------

 dbutils.library.restartPython()


# COMMAND ----------

from databricks_genai_inference import Completion

# Only required when running this example outside of a Databricks Notebook
# export DATABRICKS_HOST="https://<workspace_host>.databricks.com"
# export DATABRICKS_TOKEN="dapi-your-databricks-token"

# response = Completion.create(
#     model="databricks-meta-llama-3-70b-instruct",
#     prompt="Write 3 reasons why you should train an AI model on domain specific data sets.",
#     max_tokens=128)
# print(f"response.text:{response.text:}")


from databricks_genai_inference import ChatCompletion

# Only required when running this example outside of a Databricks Notebook
# export DATABRICKS_HOST="https://<workspace_host>.databricks.com"
# export DATABRICKS_TOKEN="dapi-your-databricks-token"

prompt = """
    What is a mixture of experts model?

    IMPORTANT: Respond only with a json document.
"""

response = ChatCompletion.create(model="databricks-meta-llama-3-70b-instruct",
                                messages=[{"role": "system", "content": "You are a helpful assistant. You only return JSON."},
                                          {"role": "user","content": prompt}],
                                max_tokens=4096)
print(f"response.message:{response.message}")

# COMMAND ----------

from databricks_genai_inference import ChatCompletion

def drug_interactions_to_kg_triples(text):
    prompt = f"""
    Task: Extract instances of drug-drug interactions from FDA-approved drug labels and other relevant sources.

    Description: A drug-drug interaction occurs when one medication (the perpetrator) alters the way the body processes another medication (the victim), leading to an unintended change in the victim medication's effects. This interaction can result in increased or decreased efficacy, or even cause adverse effects, such as toxicity or allergic reactions.

    Input (FDA-approved drug label):
    {text} 

    Output: Extracted instances of drug-drug interactions, including:
    - Perpetrator drug (the medication that affects the other drug)
    - Victim drug (the medication that is affected by the perpetrator drug)
    - Description of the interaction (e.g. increased/decreased efficacy, adverse effects)

    Example Output:
    Drug A increases risk of bleeding when taken with Drug B.
    Drug C decreased efficacy of Drug D.
    Drug E Increased risk of kidney damage when taken with Drug F.

    Instructions:
    Extract instances of drug-drug interactions from the input text.
    Identify the perpetrator drug, victim drug, and description of the interaction.

    Output the extracted information in the format specified above. Only return the extracted information.
    """

    response = ChatCompletion.create(model="databricks-meta-llama-3-70b-instruct",
                                    messages=[{"role": "system", "content": "You are a helpful assistant. "},
                                            {"role": "user","content": prompt}],
                                    max_tokens=4096)
    
    return response.message

drug_interactions_to_kg_triples(dl['drug_interactions'])

# COMMAND ----------

import json
import time
def drug_label_to_kg(drug_label):
    brand_name = drug_label['openfda']['brand_name']
    triples = drug_interactions_to_kg_triples(drug_label['drug_interactions'])
    time.sleep(1)
    return {'brand_name': brand_name,
            'drug_interactions_to_kg_triples': triples}        

    

# COMMAND ----------

def drug_labels_to_triples(drug_labels):
    clinicals = []
    for drug_label in drug_labels:
        try:
            clinicals.append(drug_label_to_kg(drug_label))
        except Exception as e:
            print(e)
            continue
    return clinicals


# COMMAND ----------

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# COMMAND ----------

t_df['clinical'] = t_df['drug_labels'].apply(drug_labels_to_triples)

t_df

# COMMAND ----------

pd.set_option('display.max_colwidth', None)
t_df

# COMMAND ----------

t_df.head(1)

# COMMAND ----------

clinical_df = t_df

# COMMAND ----------

clinical_df

# COMMAND ----------

def ddi_to_recommendation(patient_drug_list, drug_drug_interactions):
    prompt = f"""
    Task: Provide recommendations on a patient's drug regimen based on the provided patient drug regimen and the drug-drug interaction information

    Patient Drug List:
    {patient_drug_list}

    Drug Drug Interactions:
    {drug_drug_interactions}
    
    Instructions:
    Identify instances of drug-drug interactions from the Patient Drug List and Drug Drug Interaction information.
    Return those identified drug-drug interactions as one sentence written in a concise, professional format.
    If there are no identified drug-drug interactions just return 'N/A' and no other text.
    """

    response = ChatCompletion.create(model="databricks-meta-llama-3-70b-instruct",
                                    messages=[{"role": "system", "content": "You are a helpful assistant. "},
                                            {"role": "user","content": prompt}],
                                    max_tokens=4096)
    
    return response.message

# COMMAND ----------

patient_drug_list = """
Warfarin
Aspirin
Ibuprofen
"""

drug_drug_interactions = """
Aspirin increases the likelihood of patients taking Warfarin.
"""

ddi_to_recommendation(patient_drug_list, drug_drug_interactions)

# COMMAND ----------

def ddis_to_recommendations(patient_drug_list, ddis):
    recommendations = []
    print(patient_drug_list)
    
    for ddi in ddis:
        try:
            print(ddi['drug_interactions_to_kg_triples'])
            recommendation = ddi_to_recommendation(patient_drug_list, ddi['drug_interactions_to_kg_triples'])
            recommendations.append(recommendation)
            
        except Exception as e:
            print(e)
            continue
    print(recommendations)
    return recommendations




# COMMAND ----------

def drug_labels_to_drug_list(drug_labels):
    names = []
    for drug_label in drug_labels:
        if drug_label:
            brand_name = drug_label['openfda']['brand_name'][0]
            print(brand_name)
            names.append(brand_name)
    return "\n".join(names)


# COMMAND ----------

clinical_df['list_of_drugs'] = clinical_df['drug_labels'].apply(drug_labels_to_drug_list)

# COMMAND ----------

c = clinical_df[0:2]
c['recommendation'] = c.apply(lambda row: ddis_to_recommendations(row['list_of_drugs'], 
                                                                  row['clinical']), axis=1)

# COMMAND ----------

c['recommendation'].head(2)
