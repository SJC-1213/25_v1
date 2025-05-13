from flask import Flask, render_template, request, jsonify
from koni_0512_v2 import research_search
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': '검색어를 입력해주세요.'}), 400
    
    try:
        results = research_search(query)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 