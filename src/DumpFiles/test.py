import requests

try:
    response = requests.get('http://localhost:9200')
    
    # Check if the request was successful (status code 200)
    response.raise_for_status() 
    
    # Get the response data, typically in JSON format for APIs
    data = response.json() 
    print(data)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")