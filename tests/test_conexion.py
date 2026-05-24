from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("GITHUB_BASE_URL"),
    api_key=os.getenv("GITHUB_TOKEN"),
)

try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Hola Usario de samsung"
    )

    print("Conexión exitosa!")
    print(len(response.data[0].embedding))

except Exception as e:
    print(f"Error de conexión: {e}")