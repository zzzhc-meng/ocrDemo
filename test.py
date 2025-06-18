import requests

def test_ocr():
    url = "http://localhost:8000/ocr/image"
    files = {"file": open("test.jpg", "rb")}
    response = requests.post(url, files=files)
    try:
        print(response.json())
    except Exception:
        print("响应非 JSON 格式：", response.text)

if __name__ == "__main__":
    test_ocr()