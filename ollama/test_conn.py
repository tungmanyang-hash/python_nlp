import requests

OLLAMA_HOST = "http://localhost:11434"

try:
    response = requests.get(OLLAMA_HOST + "/api/tags", 
    timeout=5)
    
    if response.status_code ==200:
        print('Ollama connection successful')
    else:
        print(f"Failed to connect to Ollama. Status code:{response.status_code}")
        
except requests.exceptions.RequestException as e:
    print(f"Error connecting to Ollama: {e}")
