import boto3
import json
import logging
import os
import re
import requests
import time
import uuid
import zipfile


from knowledge_base import BedrockKnowledgeBase
from tqdm import tqdm
from urllib.parse import urlparse


logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

session = boto3.session.Session()
region = os.getenv('AWS_REGION', session.region_name)
if region:
    print(f'Region: {region}')
else:
    print("Cannot determine AWS region from `AWS_REGION` environment variable or from `boto3.session.Session().region_name`")
    raise

s3_client = boto3.client('s3', region)
bedrock_agent_client = boto3.client('bedrock-agent', region)
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region)


def download_file(url):
    destination = os.path.basename(urlparse(url).path)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        file_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def extract_zip_file(zip_path):
    try:
        n_files = 0
        with zipfile.ZipFile(zip_path, 'r') as f:
            file_list = f.namelist()
            print(f"Extracting files from {zip_path}")
            for file in tqdm(file_list, desc="Extracting"):
                if file.startswith('__') or '.DS_Store' in file:
                    print(f'Skipping file: {file}')
                    continue
                clean_filename = re.sub(r'[\s-]+', '-', file).lower()
                f.extract(file, '.')
                os.rename(file, clean_filename)
                n_files += 1
        print(f"Successfully extracted {n_files} files")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def create_s3_bucket_with_random_suffix(prefix):
    random_suffix = str(uuid.uuid4())[:8]
    bucket_name = f"{prefix.lower()}-{random_suffix.lower()}"
    try:
        if region == "us-east-1":
            # For us-east-1, we don't specify LocationConstraint
            response = s3_client.create_bucket(
                Bucket=bucket_name
            )
        else:
            # For other regions, we need to specify LocationConstraint
            response = s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': region
                }
            )

        print(f"Successfully created bucket: {bucket_name}")

        # Wait for the bucket to be available
        waiter = s3_client.get_waiter('bucket_exists')
        waiter.wait(Bucket=bucket_name)
        return bucket_name

    except Exception as e:
        print(f"Error creating bucket: {e}")
        return None

def upload_directory(path, bucket_name):
    for root,dirs,files in os.walk(path):
        for file in files:
            file_to_upload = os.path.join(root,file)
            basename = os.path.basename(file_to_upload)
            if basename == ".DS_Store":
                continue
            print(f"uploading file {file_to_upload} to {bucket_name}")
            s3_client.upload_file(file_to_upload,bucket_name,file)

def create_bedrock_knowledge_base(name, description, s3_bucket):
    knowledge_base = BedrockKnowledgeBase(
        kb_name=name,
        kb_description=description,
        data_bucket_name=s3_bucket,
        embedding_model = "amazon.titan-embed-text-v2:0"
    )
    print("Sleeping for 30 seconds.....")
    time.sleep(30)
    return knowledge_base

def ingest_knowledge_base_documents(knowledge_base_id, data_source_id, s3_bucket, kb_folder):
    # Ingest all files in folder into Bedrock Knowledge Base Custom Data Source
    kb_files = [ file for file in os.listdir(kb_folder) if file.endswith('.pdf') ]

    documents = []
    for kb_file in kb_files:
        s3_uri = f's3://{s3_bucket}/{kb_file}'
        clean_filename = re.sub(r'[\s-]+', '-', kb_file)
        custom_document_identifier = os.path.splitext(clean_filename)[0]
        custom_document_identifier = custom_document_identifier.lower()
        print(f'{s3_uri} -> Custom Document Identifier: "{custom_document_identifier}"')
        documents.append(
            {
                'content': {
                    'custom': {
                        'customDocumentIdentifier': {
                            'id': custom_document_identifier
                        },
                        's3Location': {
                            'uri': s3_uri
                        },
                        'sourceType': 'S3_LOCATION'
                    },
                    'dataSourceType': 'CUSTOM'
                }
            }
        )
    try:
        response = bedrock_agent_client.ingest_knowledge_base_documents(
            dataSourceId = data_source_id,
            documents=documents,
            knowledgeBaseId = knowledge_base_id
        )
        print(json.dumps(response, indent=2, default=str))
    except Exception as e:
        print(f'Exception: {e}')
        return None
    return response


def main():
    folder = 'pets-kb-files'
    if not os.path.isdir(folder):
        download_file('https://d3k0crbaw2nl4d.cloudfront.net/pets-kb-files.zip')
        extract_zip_file('pets-kb-files.zip')
        s3_bucket = create_s3_bucket_with_random_suffix('bedrock-kb-bucket')
        print(f'Created S3 bucket: {s3_bucket}')
        upload_directory("pets-kb-files", s3_bucket)
    else:
        print('Skipping download as folder {folder} already exists.')
        buckets = s3_client.list_buckets()['Buckets']
        s3_buckets = [ b['Name'] for b in buckets if b['Name'].startswith('bedrock-kb-bucket') ]
        s3_bucket = s3_buckets[0]

    # Create Bedrock Knowledge Base
    response = bedrock_agent_client.list_knowledge_bases()
    knowledge_bases = response.get('knowledgeBaseSummaries')
    if not len(knowledge_bases):
        random_suffix = str(uuid.uuid4())[:8]
        knowledge_base = create_bedrock_knowledge_base(
            name = f'pets-kb-{random_suffix}',
            description = 'Pets Knowledge Base on cats and dogs',
            s3_bucket = s3_bucket
        )
        knowledge_base_id = knowledge_base.get_knowledge_base_id()
        data_source_id = knowledge_base.get_datasource_id()
        print(f'Created Bedrock Knowledge Base with ID: {knowledge_base_id}')
    else:
        print('Skipping Bedrock Knowledge Base Creation')
        knowledge_base_id = knowledge_bases[0]['knowledgeBaseId']
        response = bedrock_agent_client.list_data_sources(knowledgeBaseId=knowledge_base_id)
        data_sources = response['dataSourceSummaries']
        data_source_ids = [ d['dataSourceId'] for d in data_sources ]
        if len(data_source_ids):
            data_source_id = data_source_ids[0]
        else:
            print('Error: Data source not created. Please create a custom data source manually')
            return

    print(f'Loading documents into Bedrock Knowledge Base: {knowledge_base_id}')
    print(f'Data Source ID: {data_source_id}')

    # Ingest documents from S3 into Bedrock Knowledge Base
    # Requires the appropriate S3 bucket permissions in the
    # Knowledge Base role: AmazonBedrockExecutionRoleForKnowledgeBase_xx 
    ingest_knowledge_base_documents(
        knowledge_base_id = knowledge_base_id,
        data_source_id = data_source_id,
        s3_bucket = s3_bucket,
        kb_folder = folder
    )

if __name__ == '__main__':
    main()
