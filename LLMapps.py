import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

# Disable metrics export - only traces will be sent
os.environ["TRACELOOP_TELEMETRY"] = "false"
os.environ['OTEL_METRICS_EXPORTER'] = "none"
os.environ['OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE'] = "delta"

headers = { "Authorization": "Api-Token dt0c01.Xxxxxxxx" }
Traceloop.init(
    app_name="LLM-App",
    api_endpoint="https://xxxxxxxx.live.dynatrace.com/",
    headers=headers,
    disable_batch=True
)

# openai.api_key = os.getenv("OPENAI_API_KEY")

AZURE_OPENAI_API_KEY="cxxxxxxx"
AZURE_OPENAI_ENDPOINT="https://octo-xxxxx.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT="gpt-4o-classification"
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
llm=AzureChatOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                deployment_name=AZURE_OPENAI_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=0.5,
                streaming=True,
                max_retries=2,  # Add retry logic
                timeout=30.0,   # Add timeout
                    )
@task(name="add_prompt_context")
def add_prompt_context():
    prompt = ChatPromptTemplate.from_template("explain the Gen AI capabilities in Manufacturing domain in a max of {length} words")
    model = llm
    chain = prompt | model
    return chain

@task(name="prep_prompt_chain")
def prep_prompt_chain():
    return add_prompt_context()

@workflow(name="ask_question")
def prompt_question():
    chain = prep_prompt_chain()
    return chain.invoke({"length": 50})

if  __name__ == "__main__":
    result = prompt_question()
    print(result.content)
