# C:\Users\slast\miniconda3\python.exe eda.py

import os
from dotenv import load_dotenv
import boto3

# Wczytaj dane z pliku .env
load_dotenv()

SPACES_KEY = os.getenv("SPACES_KEY")
SPACES_SECRET = os.getenv("SPACES_SECRET")
SPACES_REGION = os.getenv("SPACES_REGION")
SPACES_NAME = os.getenv("SPACES_NAME")

# Utwórz klienta S3 dla DigitalOcean Spaces
session = boto3.session.Session()
client = session.client(
    's3',
    region_name=SPACES_REGION,
    endpoint_url=f'https://{SPACES_REGION}.digitaloceanspaces.com',
    aws_access_key_id=SPACES_KEY,
    aws_secret_access_key=SPACES_SECRET
)

# Wyświetl zawartość bucketa
response = client.list_objects_v2(Bucket=SPACES_NAME)

if 'Contents' in response:
    print(f"📦 Zawartość bucketa '{SPACES_NAME}':")
    for obj in response['Contents']:
        print(f"- {obj['Key']} ({obj['Size']} bajtów)")
else:
    print("❗ Bucket jest pusty lub nie istnieje.")



from dotenv import load_dotenv
import os

# Wczytaj zmienne środowiskowe z pliku .env
load_dotenv()

# Sprawdź czy klucze są dostępne
print("SPACES_KEY:", os.getenv("SPACES_KEY")[:6], "...")  # fragment klucza
print("SPACES_SECRET:", os.getenv("SPACES_SECRET")[:6], "...")
