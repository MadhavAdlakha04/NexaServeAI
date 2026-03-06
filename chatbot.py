from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from xml.etree.ElementTree import ParseError
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import fitz  # PyMuPDF





load_dotenv()

endpoint= HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task='conversational'
)
llm = ChatHuggingFace(llm=endpoint)

#string parser
parser= StrOutputParser()


#indexing

#load pdf data 
pdf_path = "docs/Onelap Support DOC V1.pdf"
document_content = ""

with fitz.open(pdf_path) as doc:
    for page in doc:
        document_content += page.get_text()

#preview document content
# print(document_content)

#text-splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([document_content])

#embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vector_store = FAISS.from_documents(chunks, embeddings)

#retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


#augmentation

chat_history=[]

# format retrieved chunks of docs together
# def format_docs(retrieved_documents):
#   context_text= "\n\n".join(doc.page_content for doc in retrieved_documents)
#   return context_text
#-----prompts 

#system prompt
system_prompt= (
   """
      You are a helpful Customer Service Executive Assistant, named - NexaServeAI
      Answer only from the provided transcript Context
      Use the provided context from support documents to answer questions helpfully and politely.
      If the customer query is unclear or vague ask user to further clarify their request.
      If you are unsure, say you’ll escalate the issue to a human representative.
      Do not hallucinate or give made up answer and be concise.
      \n\n
      Context : {context}
"""
)

#context_aware_retriever_prompt
retriever_prompt=(
    """
        Given a chat history and the latest user question which might reference context in the chat history,
        formulate a standalone question which can be understood without the chat history.
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

        
"""
)

#context_aware_restructured_prompt_generation

contextualized_query_prompt=ChatPromptTemplate.from_messages(
   [
      ("system", retriever_prompt),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{input}")
   ]
)


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

context_adjusted_input_genreation_chain= contextualized_query_prompt | llm | parser
# context_adjusted_prompt=context_adjusted_prompt_genreation_chain.invoke(input()) 

#context_aware prompt to retrieval
context_aware_retrieval= context_adjusted_input_genreation_chain | retriever | RunnableLambda(combine_docs)


#context aware prompt template
prompt= ChatPromptTemplate.from_messages(
   [
      ("system", system_prompt),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}")
   ]
)

question_answer_chain= prompt | llm 

rag_chain = RunnableParallel({
    "context" : context_aware_retrieval,
    "input" : context_adjusted_input_genreation_chain,
    "chat_history" : RunnableLambda(lambda x: x["chat_history"]) 
}) | question_answer_chain

#TODO -> Add the query refinement for ambiguity or anything.

while True:
    customer_query = input()
    if customer_query in ('exit', 'end'):
       break;
    output = rag_chain.invoke({"input" : customer_query , "chat_history" : chat_history})
    print(output.content)
    chat_history.extend([
        HumanMessage(content=customer_query),
        output
    ])

print("CHAT HISTORY \n")
print("\nCHAT HISTORY:\n")
for msg in chat_history:
    if isinstance(msg, HumanMessage):
        print(f"You: {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"NexaServeAI: {msg.content}")
