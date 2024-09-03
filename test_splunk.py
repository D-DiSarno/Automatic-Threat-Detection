import requests
import logging

logging.basicConfig(level=logging.DEBUG)

def test_send_to_splunk():
    splunk_url = "http://localhost:8088/services/collector/event"
    splunk_token = ""
    headers = {'Authorization': f'Splunk {splunk_token}'}
    data = {
        "event": {"test": "test_data"},
        "sourcetype": "json"
    }
    response = requests.post(splunk_url, headers=headers, json=data)
    if response.status_code == 200:
        logging.debug("Test results sent to Splunk successfully.")
        print("Test results sent to Splunk successfully.")
    else:
        logging.error(f"Failed to send test results to Splunk: {response.text}")
        print(f"Failed to send test results to Splunk: {response.text}")

if __name__ == "__main__":
    test_send_to_splunk()
