import os

import numpy as np
import google.generativeai as genai
from google.generativeai.types import content_types
from flask import Flask, request, jsonify, make_response
from dotenv import load_dotenv
from datasets import load_from_disk
from flask_cors import CORS
from markdown import markdown

from tools import get_event_data, display_map, search_youtube_video

user_chat_sessions = {}

load_dotenv()

events_data = load_from_disk('events_data')
events_data.add_faiss_index(column="embeddings")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_event(query):
    global events_data
    query = np.array(genai.embed_content(model="models/text-embedding-004", content=query)['embedding'])
    score, sample = events_data.get_nearest_examples('embeddings', query, k=1)
    return sample['stringify']

def format_message(role, parts):
    return {'role': role, 'parts': parts}

def get_system_instruction():
    system_instruction = (
        '너는 외로운 서울 시민에게 개인 맞춤 컨텐츠를 추천하는 모델이야. '
        '여기서 말하는 컨텐츠는 서울시에서 하는 문화행사, 유튜브 비디오들이 있어. '
        '대화를 통해 상대방을 파악하고 이에 따라서 상대방에게 능동적으로 컨텐츠를 소개해줘. '
        '상대방이 컨텐츠를 요구했을 때는 추가적인 재질문 없이 먼저 컨텐츠를 소개해줘. '
        '왜냐하면 너가 먼저 추천한 컨텐츠에 대해 상대방이 호불호를 말해줄 수 있기 때문이야 '
        '상대방이 명시적으로 컨텐츠에 대한 추천을 요구하지 않았다고 하더라도 상대방의 상황이나 감정에 맞는 컨텐츠를 소개해줘 '
        '너의 궁극적인 목적은 외로운 서울 시민에게 도움이 되는 것이야. 따듯하고 자상한 말투로 상대방을 대해줘. '
        '상대방이 특정 지역에 대해서 궁금해하거나 특정 지점으로 가는 경로를 궁금해하면 display_map 함수를 통해서 지도를 사용할 수 있어. '
        '이런 상황에서는 어떻게 갈 것인지 가서 무엇을 할 것인지 재차 물어보지 말고 바로 dispaly_map 함수를 호출해줘 '
        '요새 서울에서 진행하는 문화행사는 get_event_data 함수를 호출하여 얻어올 수 있어 이를 유용하게 사용해줘 '
        'search_youtube_video 함수를 사용하면 사용자가 관심있어할만한 유튜브 비디오를 보여줄 수 있어 '
    )
    return format_message('system', system_instruction)

tools = [get_event_data, display_map, search_youtube_video]
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
generation_config = genai.GenerationConfig(temperature=0)
model = genai.GenerativeModel(os.environ['GEMINI_TEXT_GENERATION_MODEL'], system_instruction=get_system_instruction(), tools=tools, generation_config=generation_config)
#model = genai.GenerativeModel(os.environ['GEMINI_TEXT_GENERATION_MODEL'], system_instruction=get_system_instruction(), tools=tools)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://itda.seoul.kr", "http://localhost:3000"]}}, supports_credentials=True)

@app.route('/', methods=["GET"])
def hello():
    return "Hello, Flask!"

# curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"name": "John", "query": "청량리 근처에 놀 수 있는 곳이 있어?"}'
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # 요청에서 JSON 데이터 받기
    name = request.cookies.get('name')
    if name not in user_chat_sessions:
        user_chat_sessions[name] = model.start_chat(history=[])
    if 'query' not in data or len(data['query'].strip()) == 0:
        return jsonify({"error": "Please enter something!"}), 400
    query = data['query']
    responses = []
    if '카페' in query:
        response = user_chat_sessions[name].send_message(query, tool_config=content_types.to_tool_config({"function_calling_config": {"mode": "any", "allowed_function_names": 'google_search_retrieval'}}))
    else:
        response = user_chat_sessions[name].send_message(query)
    response_parts = []
    for part in response.parts:
        if fn := part.function_call:
            fn_name = fn.name
            if fn_name == 'get_event_data':
                print('event_data')
                event = get_event(query)
                response = user_chat_sessions[name].send_message(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn_name, response={'result': event}))).text
                responses.append({'text': markdown(response)})
            elif fn_name == 'display_map':
                print('map')
                responses.append({'map': fn.args['target_location']})
            elif fn_name == 'search_youtube_video':
                print('youtube')
                responses.append({'youtube': fn.args['query']})
        else:
            text = part.text
            responses.append({'text': markdown(text)})
    return jsonify({'responses': responses})

@app.route('/cookie', methods=['POST'])
def set_cookie():
    data = request.get_json()

    name = data.get("name") if data else None
    
    if not name:
        return jsonify({"error": "name 값이 제공되지 않았습니다."}), 400

    response = make_response(jsonify({"message": "쿠키에 이름이 설정되었습니다!"}))
    response.set_cookie("name", name, max_age=60*60*24)
    return response

@app.route('/cookie', methods=['GET'])
def get_cookie():
    name = request.cookies.get("name")
    if name:
        return jsonify({"message": f"쿠키에 저장된 이름은 {name}입니다."})
    else:
        return jsonify({"message": "쿠키가 설정되어 있지 않습니다."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
