import requests

def main():
    url = "http://127.0.0.1:8000/predict"
    data = {"reddit_comment": "This is a test comment to classify"}
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())

if __name__ == "__main__":
    main()
