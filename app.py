from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from dotenv import load_dotenv
import os
import boto3
import tempfile
import b2sdk.v2 as b2

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize the OpenAI API key
openai.api_key = os.getenv("api_key")

# Backblaze B2 credentials
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
BUCKET_NAME = "contentscoring"  

# Initialize B2 client
info = b2.InMemoryAccountInfo()
b2_api = b2.B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APPLICATION_KEY)
bucket = b2_api.get_bucket_by_name(BUCKET_NAME)

# Download file from B2 if it does not exist locally
def download_from_b2(file_name, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {file_name} from Backblaze B2...")
        bucket.download_file_by_name(file_name).save_to(local_path)

def download_folder_from_b2(folder_name, local_path):
    """
    Downloads all files from a Backblaze B2 folder and subfolders,
    and saves them in the corresponding local directory structure.
    
    :param folder_name: Name of the folder in Backblaze B2
    :param local_path: Local directory to save the files and subfolders
    """
    # Check if the local path exists, if not, create it
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # List all files and folders in the specified folder on Backblaze
    files = bucket.ls(folder_name)

    for file_info in files:
        file_version = file_info[0]  # FileVersion object
        file_name_str = file_version.file_name  # Access the actual file name

        # Check if the item is a folder by checking if the name ends with '/'
        if file_name_str.endswith('/'):
            # It's a folder, recursively call download_folder_from_b2
            subfolder_path = os.path.join(local_path, file_name_str[len(folder_name):].lstrip('/'))
            download_folder_from_b2(file_name_str, subfolder_path)
        else:
            # It's a file, download it to the corresponding local path
            local_file_path = os.path.join(local_path, file_name_str[len(folder_name):].lstrip('/'))  # Adjust path by removing folder_name prefix
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))  # Create the folder structure
            print(f"Downloading {file_name_str} from Backblaze B2 to {local_file_path}...")
            bucket.download_file_by_name(file_name_str).save_to(local_file_path)


# Model class for embeddings
class Model:
    def __init__(self):
        self.model_dict = None
        self._load_model()

    def _load_model(self):
        # Download model files from Backblaze
        local_path = os.path.join(os.getcwd(), 'sent_bert_mini_models')  # Local path where the folder should be downloaded
        download_folder_from_b2('sent_bert_mini_models', local_path)
        #download_folder_from_b2('sent_bert_mini_models', os.path.join(os.getcwd(), 'sent_bert_mini_models'))
        # Load the sentence transformer model
        self.model_dict = {'miniBert': SentenceTransformer('./sent_bert_mini_models')}
    
    def get_embedding_miniBert(self, text):
        # Get mini bert embedding
        embeddings = self.model_dict['miniBert'].encode([text])
        return embeddings

model = Model()

# Function to generate time-based features from date
def generate_time_features(df, timestamp_col):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True)
    df['hour_of_day'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['is_weekend'] = (df[timestamp_col].dt.dayofweek >= 5).astype(int)
    return df

# Processor function to preprocess the input data
def processor(input_data):
    input_data['date'] = pd.to_datetime(input_data['date'])
    input_data = pd.DataFrame([input_data])
    input_data = generate_time_features(input_data, 'date')
    input_data.drop(['date'], axis=1, inplace=True)
    
    input_data['content'] = input_data['subject'] + '\n' + input_data['content']
    input_data.drop(['subject'], axis=1, inplace=True)
    input_data['embeddings'] = input_data['content'].apply(lambda x: model.get_embedding_miniBert(x))
    input_data['embeddings'] = input_data['embeddings'].apply(lambda x: x.squeeze())
    
    embeddings_df = pd.DataFrame(input_data['embeddings'].tolist(), index=input_data.index)
    embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_df.shape[1])]
    
    input_data = pd.concat([input_data.drop(columns=['embeddings']), embeddings_df], axis=1)
    input_data.drop(['content'], axis=1, inplace=True)
    
    return input_data

# Scorer method to calculate score
def scorer(input_data):
    df = processor(input_data)
    # Download model if not available locally
    download_from_b2('model.joblib', './model.joblib')
    rf_model = joblib.load('./model.joblib')
    score = rf_model.predict_proba(df)[0][0]
    return score

# Class for scoring input data
class ScorerInput(BaseModel):
    subject: str
    content: str
    date: str

class SuggestionsInput(BaseModel):
    target_title: str
    target_name: str
    target_industry: str
    core_message: str

# Function to get email suggestions from OpenAI API
def send_email_function_call(title, industry, core_message, name):
    function_call = {
        "name": "linkedin_outreach"
    }
    
    functions = [
        {
            "name": "linkedin_outreach",
            "description": "LinkedIn outreach message",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Subject of the message"},
                    "content": {"type": "string", "description": "Content of the message"}
                },
                "required": ["subject", "content"]
            }
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[{
            "role": "user",
            "content": f"I want to send a message to {name} on LinkedIn. He is a {title} in the {industry} industry. The core message to convey is {core_message}. Generate three suggestions for subject and content for this."
        }],
        functions=functions,
        function_call=function_call
    )
    
    return response.choices[0].message["function_call"]["arguments"]

# API endpoint for Scorer
@app.post("/scorer")
async def scorer_endpoint(input_data: ScorerInput):
    score = scorer(input_data.dict())
    return {"score": score}

# API endpoint for Suggestions
@app.post("/suggestions")
async def suggestions_endpoint(input_data: SuggestionsInput):
    suggestions = []
    for _ in range(3):  # Generate 3 suggestions
        try:
            email_details = send_email_function_call(input_data.target_title, input_data.target_industry, input_data.core_message, input_data.target_name)
            suggestions.append(email_details)
        except Exception as e:
            suggestions.append({"error": str(e)})
    
    return {"suggestions": suggestions}
