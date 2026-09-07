import requests


data = {
    "title": "Software Engineer",

    "description": "Develop backend applications using Python and Java",
    "function": "AI Engineer",
    "industry": "Robotics",
    "location": "Mars"
    }


response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=data
)


print("Status code:", response.status_code)

print("Response:")
print(response.text)