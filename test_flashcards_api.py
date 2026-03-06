import requests
try:
    session = requests.Session()
    # First hit an endpoint to set the session cookie
    session.get('http://127.0.0.1:5000/dashboard')
    
    # Now hit the generate flashcards API
    response = session.post('http://127.0.0.1:5000/api/generate-flashcards', json={})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Script Error: {e}")
