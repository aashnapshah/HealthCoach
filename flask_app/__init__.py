from flask import Flask
from flask_app.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    from flask_app.routes import chat_bp
    app.register_blueprint(chat_bp)

    return app
