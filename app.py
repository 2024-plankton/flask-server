from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/', methods=["GET"])
def hello():
    return "Hello, Flask!"

# curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"name": "John"}'
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # 요청에서 JSON 데이터 받기
    if not data or 'name' not in data:
        return jsonify({"error": "No name provided!"}), 400
    name = data['name']
    return jsonify({"message": f"Hello, {name}!"})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
