import requests

import json

data = {
    "age": 45,
    "workclass": "Private",
    "fnlgt": 280464,
    "education": "Doctorate",
    "education_num": 10,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 510,
    "capital_loss": 0,
    "hours_per_week": 60,
    "native_country": "United-States"
}

response = requests.post('https://project3-census.herokuapp.com/inference', data=json.dumps(data), headers={'content-type': 'application/json'})

print(response.status_code)
print(response.json())
print("Live test completed")


