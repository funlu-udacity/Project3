import requests

import json

data = {
  "age": 42,
  "workclass": "Private",
  "fnlgt": 52789,
  "education": "Masters",
  "education_num": 17,
  "marital_status": "Married",
  "occupation": "Data-Scientist",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 4174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}

response = requests.post('https://project3-census.herokuapp.com/predict', data=json.dumps(data), headers={'content-type': 'application/json'})

print("Status Code:", response.status_code)
print("Prediction: ", response.json())
print("Live test completed")


