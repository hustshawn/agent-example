# import os

# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI

# chat = ChatOpenAI(
#     model="anthropic.claude-3-sonnet-20240229-v1:0",
#     temperature=0,
#     openai_api_key=os.environ['OPENAI_API_KEY'],
#     openai_api_base=os.environ['OPENAI_BASE_URL'],
# )

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate.from_template(template)
# llm_chain = LLMChain(prompt=prompt, llm=chat)

# question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
# response = llm_chain.invoke(question)
# print(response)
# print(response['text'])

from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
import os
from prompts import context, code_parser_template
from code_reader import code_reader
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
import ast

load_dotenv()

# llm = Ollama(model="llama3", request_timeout=30)
llm = Ollama(model="mistral", request_timeout=30)

# print(os.getenv("LLAMA_CLOUD_API_KEY"))
parser = LlamaParse(result_type="markdown", api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vectore_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

quey_engine = vectore_index.as_query_engine(llm=llm)

# result = quey_engine.query("What are some of the routes in the api?")
# print(result)

tools = [
    QueryEngineTool(
        query_engine=quey_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This gives the code to the documentation for an API. Use this for reading docs for the API.",
        ),
    ),
    code_reader,
]

code_llm = Ollama(model="codellama")
agent = ReActAgent.from_tools(tools, llm=code_llm, context=context, verbose=True)


class CodeOuptut(BaseModel):
    code: str
    description: str
    filename: str


parser = PydanticOutputParser(CodeOuptut)
json_prompt_tempate = parser.format(code_parser_template)
json_prompt_str = PromptTemplate(json_prompt_tempate)
output_pipeline = QueryPipeline(chain=[json_prompt_str, llm])

while (prompt := input("Enter a prompt (q to quite): ")) != "q":
    result = agent.query(prompt)
    next_result = output_pipeline.run(result=result)
    cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))

    print("Code Generated:")
    print(cleaned_json["code"])

    print("\n\nDescription", cleaned_json["descritption"])

    filename = cleaned_json["filename"]
    print(f"Filename: {filename}")
