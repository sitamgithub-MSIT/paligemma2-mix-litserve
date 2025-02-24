import requests
from rich.console import Console
from rich.markdown import Markdown

url = "http://localhost:8000/predict"

# Input image path for the test
image_path = "images/handwriting.jpg"

# Create the payload for the request
payload = {"image_path": image_path}

# Send the request to the server
response = requests.post(url, json=payload)

# Parse the response JSON data
output = response.json()

# Display the response in a markdown format
console = Console()
markdown = Markdown(output["text"])
console.print(markdown)
