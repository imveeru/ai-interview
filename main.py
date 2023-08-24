import streamlit as st

st.set_page_config(
    page_icon="🧑‍💼",
    page_title="AI Interviewer",
    initial_sidebar_state="expanded"
)

hide_st_style ='''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🧑‍💼AI Interviewer")

import json
import google.generativeai as palm
from google.auth import credentials
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel

##################### Vertext AI & PaLM API initialization #####################

# config = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"] #for deploying
config ='''{
  "type": "service_account",
  "project_id": "optimal-route-suggestion",
  "private_key_id": "19b0edee0b5acd92fd2068e84dec0c68d06eac6c",
  "private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCdBicLKZ6Bs/wj\\nAdo9UNmvPRE2dL3cZS3uIQeK4tx0Vvy3l0b56HdoLu0QMwTWgOWe+YWXfAUwGB7q\\n+0pm8w1+MOkqC0V2IBVJl5/c4CEx3u+24kVHi6bnC90IgReJwC0xwN1HaCJatESA\\n4fCJYnN1qRKXU1gx3KFGxl9UX3+sWG+TPoWATVfKlf5poLrzZJ9E9QglkMl+slpm\\nRjdb3M4f2pFmL1/190bilxfokLT3+LqC3GaIIPQY7lWlWACXt+ldvqsTiIMo/z+Z\\nGwoSJ4xwd+O3JtUHhh+KSRY6Oc43OIhJhKcqsXsuuRl4LWu//uImrt2oilcXguYU\\nJRSrvU8bAgMBAAECggEARCGrRzijwftqZ3YiT4CJM3P3x/0XdE2ihDRopWaR6Rjl\\nRnOpJD4tsVLLIcBBVSFQgI4b3QK+7YNJxwOJ4OmM7Tgjs054sSxykB/uCVRmktD8\\nignbrZN2s8F+Anag0/BCq9fXK2iPn3OgVZuzVqkVF/RoUKilF913TNI+Asn9B7YZ\\nkeoUqSNUUN5fjIgKNWMEvolKZu+TdYobxV9GOevX8FwTx87qmF9Ue8P1RxzyAD3X\\nMBjm5ohZufg6zctN0vxK2OXGvV9lc/qfsxtHhtRoyS3F0Da2NZIXSndl3Arh0Bcb\\nq8b9eo85yjcy+oGVwo00gytpJBk2k445EtwWW0p7oQKBgQDLU5Ux9N9hCT6p1f4N\\n0ngaihCxjxnGu4ZPR3tIiqG5JWN+0vr0obVr9A5m3LU7/AG9Bmm85MlQi6FmuoDw\\neqQY85nImx5z3i55CTx9HOOTpA2pbTIHNqZ8aPLxZI0he1Pnvu6qpiRH3d51Nby4\\nJ4Qbi+Wd+CmAy6jHib3yNL+eTQKBgQDFs9XGv/gfsparX1DMoAfU1v97WkXxUruI\\nxCmSlUpVb4kmDs4gk8x9GJcokvxEr5+iJUPkzwexW/LBaulAdJB3dR8phvGBqvag\\nMe04IkeKVuOXwTgM1QMZmbIvLL3fA546dgksXK5vZ5qal+N3GK/dLhIPQk7zo1nO\\nuy/Y5otnBwKBgQCjCBj4HpXCU8xYF8sGwD0nYo8yIEEV1aVDCljy+J3mO/GEbp1k\\n7AjxT5cAqXX0bAPk0jCUkopNODipi1/58wyDKUikzqRjWcK/sEU9OJ3N81w0/uZ/\\nXDWwSeKK5go3z5CeoLz0PhWXPnKyXu08aAsIn2r0+Fgm+qYRoQOaIuuGfQKBgCWy\\nNIDA+b6RfskOU4mwuc2LcQtEGzH4ZGmffY3FiXbg3XW0PPlZNRRlK+1AmXk/Q2DX\\nWiq2jvDyZ0cZ63+uuh0M5/QzFrlyr7O70U9yudFW3+5/mQBZXU30UFVOYqWzOuhK\\nuVUMFvaG+qOfcm+y9VVnA2qFaihqbSVN68Gfs9ThAoGAacSC7rPBXrYbWJaHpwcc\\nb0b/AODiz03HS/YPWpG778mJXueH7l05RYMCHmfQlLsCUia7j87MaVppok7lb7HY\\nH0fPt0y9fmPB/YT3rX/jsFrBDzuJynd2pYZqgjwWLjWfMRjlg1SdOpzdeK81yVCw\\nHCs87atWEc87lcaIVCzItbg=\\n-----END PRIVATE KEY-----\\n",
  "client_email": "langchainapps@optimal-route-suggestion.iam.gserviceaccount.com",
  "client_id": "103157086402886138790",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/langchainapps%40optimal-route-suggestion.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
''' #for testing
service_account_info=json.loads(config)
service_account_info["private_key"]=service_account_info["private_key"].replace("\\n","\n")
# st.write(service_account_info)
my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)
aiplatform.init(
    credentials=my_credentials,
)
project_id = service_account_info["project_id"]
vertexai.init(project=project_id, location="us-central1")

parameters = {
    "max_output_tokens": 1024,
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison@001")

from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

##################### LLM #####################

llm=ChatVertexAI()
memory=ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100
)
conversation=ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

def get_prompt(position):
    return f'''
        You are a panelist for a mock interview session. 
        Your task is to conduct an interview with a candidate for a specific position.
        The position you will be interviewing the candidate for is {position}.
        During the interview, you should ask a combination of general questions to assess the candidate's overall suitability and role-specific questions to evaluate their qualifications for the position. 
        After each response, provide constructive feedback on the candidate's answers, highlighting strengths and areas for improvement.
        Focus on maintaining a professional and objective approach throughout the interview process.
        **You must only reply as the interviewer. DO NOT WRITE ALL THE CONVERSATION AT ONCE**
        Start the interview with a greeting and getting to know about the candidate. Do not mention name in greetings, make it generalized one.
        **I want you to only do the interview with me. Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers. **
        Conduct the mock interview and provide detailed feedback including a prediction on possibility of crcking the job interview based on the candidate's responses.
    '''

# res=conversation.predict(input="Hello, I'm Jack!")
# st.write(res)

# mem=memory.load_memory_variables({})
# st.divider()
# st.write(mem)

##################### UI #####################

with st.sidebar:
    position=st.text_input("What position do you wish to be interviewed?")
    if position:
        prompt=get_prompt(position)
    
if position and prompt:
    with st.spinner("Get ready for the interview..."):
        res=conversation.predict(input=prompt)
        st.write(res)