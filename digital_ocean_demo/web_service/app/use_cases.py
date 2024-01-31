from typing import List, Tuple

import torch
import numpy as np
from transformers import pipeline
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers.models.dpr.modeling_dpr import (
    DPRQuestionEncoderOutput,
    DPRContextEncoderOutput,
)

from app.documents import documents, generated_document


question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)


class AnswerQuestionUseCase:
    def __init__(self, question: str) -> None:
        self._question = question

    def execute(self) -> str:
        # Generate response without information
        content = "Write a paragraph that answers the question"
        # generated_document = intruct_completion(self._question, content)  # CUrrently commented for time restriction
        # Obtain embedding from question and from generated document
        (
            question_embeddings,
            generated_document_embedding,
            document_embeddings,
        ) = get_embeddings(self._question, generated_document)
        document_embeddings = document_embeddings.pooler_output[0]
        generated_document_embedding = generated_document_embedding.pooler_output[0]
        print(f"type(document_embeddings): {type(document_embeddings)}")
        print(f"type(generated_document_embedding): {type(generated_document_embedding)}")
        # Obtain most similar document
        best_index = None
        best_dist = None
        for i in range(len(documents)):
            document_embedding = document_embeddings[i]
            dist = cosine_similarity(document_embedding.detach().numpy(), generated_document_embedding.detach().numpy())
            if best_dist is None or dist > best_dist:
                best_dist = dist
                best_index = i
        # insert most similar document into context
        content = f"Write a paragraph that answers the question using the following information:\n{documents[best_index]}"
        print(f"best_index: {best_index}")
        # answer question
        return self._question


def intruct_completion(question: str, content: str) -> str:
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-alpha",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [
        {
            "role": "system",
            "content": content,
        },
        {"role": "user", "content": question},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs[0]["generated_text"]


def get_embeddings(
    question: str, document: str
) -> Tuple[
    DPRQuestionEncoderOutput, DPRContextEncoderOutput, DPRContextEncoderOutput
]:
    question_embeddings = encode_queries([question])
    generated_document_embedding = encode_documents([document])
    document_embeddings = encode_documents(documents)
    return (
        question_embeddings,
        generated_document_embedding,
        document_embeddings,
    )


def cosine_similarity(embedding1, embedding2):
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    print(f"type(similarity): {type(similarity)}")
    return similarity


def encode_queries(queries: List[str]):
    return question_encoder(
        **question_tokenizer(
            queries, return_tensors="pt", padding=True, truncation=True
        )
    )


def encode_documents(documents: List[str]):
    return context_encoder(
        **context_tokenizer(
            documents, return_tensors="pt", padding=True, truncation=True
        )
    )
