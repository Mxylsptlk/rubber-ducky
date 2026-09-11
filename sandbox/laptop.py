# %%
import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.getenv('HOME') +'/Documents/git/rubber-ducky'
load_dotenv(os.path.join(BASE_DIR,'.env'))

# %%


sys.path.append(BASE_DIR + '/rubber_ducky')

from agent_graph import RubberDuckyAgentGraph

# %%

from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings

model = init_chat_model(
    "claude-sonnet-5"
)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

graph = RubberDuckyAgentGraph(model, embeddings)

# %%

result = graph.invoke(
	("Give me some ideas for finding leads for a solopreneur "
	  "working in analytics and predictive modeling.")
)

# %%

