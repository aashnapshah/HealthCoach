from flask import Blueprint, render_template, request, jsonify
from utils.runpod_client import send_to_runpod

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/')
def chat():
    return render_template('chat.html')

@chat_bp.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message')
    context = data.get('context')
    
    try:
        response = send_to_runpod(message, context)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
