import os

import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datasets import load_from_disk
from flask_cors import CORS

from tools import get_event_data, display_path

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
        '당신은 외로운 서울 시민에게 개인 맞춤 외출 가이드 서비스를 제공하는 심리상담가입니다. '
        '당신은 서울 시민의 외출 요청에 대해 가이드를 제공하고, 서울 시민과의 대화를 통해 서울 시민의 외출 요청에 대한 정보를 얻습니다. '
        '대화를 통해 얻은 정보를 바탕으로 사용자에게 꼭 맞는 외출 장소를 추천해주세요. '
        '추가적으로 당신은 현재 서울에서 열리는 문화행사에 대한 데이터를 접근 할 수 있습니다. '
        '사용자가 현재 열리는 문화행사에 대해서 질문하거나 그와 유사한 질문을 하면 get_event_data 함수를 호출하여 현재 진행되는 이벤트를 확인하세요. '
        '사용자가 특정 지역에 가는 길을 물어보거나 특정 지역에 대해서 궁금해한다면 display_path 함수를 사용하여 그 지역에 대한 정보를 알려줘 '
        '이 질문을 받았을 때, 출발 지점이나 가서 뭐 할지는 고려 대상이 아니야. '
        '이 명령을 받으면 일단 먼저 display_path 함수를 호출해줘. 상대적으로 우선순위가 높다고 볼 수 있어. '
        'display_path의 argument type은 string이야. 절대 proto.marshal.collections.maps.MapComposite이 아니야 '
    )
    return format_message('system', system_instruction)

tools = [get_event_data, display_path]
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel(os.environ['GEMINI_TEXT_GENERATION_MODEL'], system_instruction=get_system_instruction(), tools=tools)

app = Flask(__name__)
CORS(app)

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
    responses = []
    response = user_chat_sessions[name].send_message(query)
    response_parts = []
    for part in response.parts:
        if fn := part.function_call:
            fn_name = fn.name
            if fn_name == 'get_event_data':
                event = get_event(query)
                response = user_chat_sessions[name].send_message(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn_name, response={'result': event}))).text
                responses.append({'text': response})
            elif fn_name == 'display_path':
                print('map')
                print(fn.args)
                breakpoint()
                exit()
                pass # implement this
        else:
            text = part.text
            responses.append({'text': text})
    return jsonify({'responses': responses})

    # response = st.session_state.chat.send_message(prompt)
    # response_parts = []
    # for part in response.parts:
    #     if fn := part.function_call:
    #         fn_name = fn.name
    #         if fn_name == 'display_path':
    #             function_response = display_path(**fn.args)
    #         #print(function_response)
    #         response_parts.append(genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn_name, response={"result": function_response})))
    #         # send it to the front
    #         #st.session_state.chat.send_message(response_parts)
    #     else:
    #         text = part.text
    #         st.session_state.messages.append({'role': 'model', 'parts': text})
    #         st.chat_message('model', avatar='static/img/therapist.jpg').write(text)
    # if response_parts:
    #     response = st.session_state.chat.send_message(response_parts).text
    #     st.session_state.messages.append({'role': 'model', 'parts': text})
    #     st.chat_message('model', avatar='static/img/therapist.jpg').write(text)
    # st.session_state.messages.append({'role': 'model', 'parts': response})
    # st.chat_message('model', avatar='static/img/therapist.jpg').write(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
