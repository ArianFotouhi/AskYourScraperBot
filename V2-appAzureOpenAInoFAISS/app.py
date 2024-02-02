import requests
from bs4 import BeautifulSoup
import json
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
import warnings
warnings.filterwarnings('ignore')

with open(r'../config/config.json') as config_file:
    config_details = json.load(config_file)
    
# e.g. "2023-07-01-preview"
openai_api_version= config_details["API_VERSION"]
# create one and call it here e.g. "myGPT"
deployment_name= config_details["DEPLOYMENT_NAME"]
# e.g. https://X.openai.azure.com/
openai_api_base= config_details["OPENAI_API_BASE"]

openai_api_key = config_details["OPENAI_API_KEY"]
openai_api_type="azure"

def scrape_web_page(urls):
    text = ''
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text += soup.text  + "\n\n" 

    return text
    


def extract_answers(question, text):
    
    model  = AzureChatOpenAI(
        openai_api_base=openai_api_base,
        openai_api_version=openai_api_version,
        deployment_name=deployment_name,
        openai_api_key=openai_api_key,
        openai_api_type=openai_api_type,
    )

    content = f""" My question is {question}
                    Answer it based on the below information:
                    {text}
                """
    message = HumanMessage(
                  content=content
                        )

    response = model([message])

    # Generate an answer

    return response.content


def handle_user_input(url, question):
    # Scrape the web page
    scraped_text = scrape_web_page(url)
    print(scraped_text)

    if scraped_text:
        # Ask the user for a question
        # Extract the answer
        answer = extract_answers(question, scraped_text)

        # Print the answer
        print(f"Answer: {answer}")
    else:
        print("Failed to scrape the web page.")

# Test the application
urls = [
    "https://github.com/ArianFotouhi",
        "https://www.bbc.com/sport/football/premier-league/top-scorers",


]
question = 'What do you know about arian fotouhi?'

handle_user_input(urls, question)