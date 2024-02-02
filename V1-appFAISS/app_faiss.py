import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os
from langchain.llms import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''
os.environ["OPENAI_API_KEY"] = ""


###########################################################################Scraper
url_directories = [
    "https://en.wikipedia.org/wiki/Candiac,_Quebec",
]

# Variable to store the extracted text
text = ""

# Iterate through the URLs and extract text
for url in url_directories:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for any HTTP errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the text from the HTML
        page_text = soup.get_text()
        text += page_text  # Append the text to the variable

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")


###########################################################################LLM
load_dotenv()

# Print or do something with the extracted text
text_splitter = CharacterTextSplitter(
    separator="\n", #new line is used to split the text
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text)

# create embedding
embeddings = HuggingFaceEmbeddings()

knowledge_base = FAISS.from_texts(chunks,embeddings)


while True:
    user_question = input("Ask Me: ")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        # repo_id = "google/flan-t5-xxl"
        repo_id = "google/flan-t5-large"
        llm = HuggingFaceHub(repo_id= repo_id, model_kwargs={"temperature":0.5, "max_length":64})

        
        # llm = ChatOpenAI( openai_api_key= os.getenv("OPENAI_API_KEY"), temperature=0, model_name="gpt-3.5-turbo")

        chain = load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents=docs,question=user_question)
        print('Response:', response)
