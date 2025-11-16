import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url': 'https://raw.githubusercontent.com/katwre/Immune-cell-classifier-api/main/images/erythroblast_example.png'}

result = requests.post(url, json=data).json()
print(result)
