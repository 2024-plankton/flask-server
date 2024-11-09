import os

import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

user_chat_sessions = {}

load_dotenv()

def format_message(role, parts):
    return {'role': role, 'parts': parts}

def get_system_instruction():
    system_instruction = (
        '당신은 외로운 서울 시민에게 개인 맞춤 외출 가이드 서비스를 제공하는 심리상담가입니다. '
        '당신은 서울 시민의 외출 요청에 대해 가이드를 제공하고, 서울 시민과의 대화를 통해 서울 시민의 외출 요청에 대한 정보를 얻습니다. '
        '대화를 통해 얻은 정보를 바탕으로 사용자에게 꼭 맞는 외출 장소를 추천해주세요. '
    )
    return format_message('system', system_instruction)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel(os.environ['GEMINI_TEXT_GENERATION_MODEL'], system_instruction=get_system_instruction())

app = Flask(__name__)

@app.route('/', methods=["GET"])
def hello():
    return "Hello, Flask!"

# curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"name": "John", "query": "청량리 근처에 놀 수 있는 곳이 있어?"}'
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # 요청에서 JSON 데이터 받기
    if not data or 'name' not in data:
        return jsonify({"error": "No name provided!"}), 400
    name = data['name']
    if name not in user_chat_sessions:
        user_chat_sessions[name] = model.start_chat(history=[])
    if 'query' not in data or len(data['query'].strip()) == 0:
        return jsonify({"error": "Please enter something!"}), 400
    query = data['query']
    gemini_response = user_chat_sessions[name].send_message(query).text
    return jsonify({'response': gemini_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
