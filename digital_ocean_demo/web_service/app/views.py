from flask import Blueprint, request

from app.use_cases import AnswerQuestionUseCase


bp = Blueprint("routes", __name__)


@bp.route("/hello-world", methods=["GET"])
def hello_world():
    return "Hola Mundo!", 200


@bp.route("/answer_question", methods=["POST"])
def answer_question():
    question = request.json.get('question')
    question = "what does physical medicine do?"  # Overwrite user question for demo
    answer = AnswerQuestionUseCase(question).execute()
    return {"result": {"answer": answer}}, 200
