import requests

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, json=payload)
        print("Status:", response.status_code, response.text)
        return response.status_code == 200
    except Exception as e:
        print("Error sending Telegram message:", e)
        return False
