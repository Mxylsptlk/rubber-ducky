# %%
import os
import sys
from dotenv import load_dotenv

BASE_DIR = '/home/phil/git/rubber-ducky'
load_dotenv(os.path.join(BASE_DIR,'.env'))

# %%


sys.path.append(BASE_DIR + '/rubber_ducky')

from agent_graph import RubberDuckyAgentGraph

# %%

from langchain_ollama import OllamaLLM, OllamaEmbeddings

model = OllamaLLM(model="llama3.2:3b")
brainstormer = OllamaLLM(model="gemma4:26b")
problem_solver = OllamaLLM(model="gemma4:26b")

embeddings = OllamaEmbeddings(model='nomic-embed-text')

graph = RubberDuckyAgentGraph(
    model,
    embeddings,
    # brainstorming_model=brainstormer,
    # problem_solving_model=problem_solver
)

# %%

result = graph.invoke(
	("Give me some ideas for finding leads for a solopreneur "
	  "working in analytics and predictive modeling.")
)

print(result['response'])
print(result['route'])

# %%

result = graph.invoke(
	("My cold outreach is failing. How do I get responses?")
)

print(result['response'],'\n\n','routing: '+result['route'])

# %%

# %%

