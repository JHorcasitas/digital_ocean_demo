from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    from app.views import bp
    app.register_blueprint(bp)

    return app