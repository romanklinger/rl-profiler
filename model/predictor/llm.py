import random

from transformers import AutoTokenizer, pipeline
import torch


class LLM():
    def __init__(self,
                 args,
                 temperature=0.8,
                 top_p=0.9,
                 top_k=40,
                 max_length=2048):

        self.args = args
        self.CLASS_NAMES = args.class_names
        self.model = args.inference_model_name

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, token=True)
        self.llama_pipeline = pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="balanced"
        )

    def get_response(self, prompt: str) -> str:
        sequences = self.llama_pipeline(
            prompt,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=100,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = sequences[0]["generated_text"]  # type: ignore
        response = response.split("[/INST]", 1)[1]
        response = response.replace("\n", "")
        response = response.lower()
        return response

    def evaluate_author(self,
                        selected_tweets,
                        y_true,
                        author_id=0,
                        PRINT_LLM_INPUT_OUTPUT=False):
        formatted_tweets = ""
        for i, tweet in enumerate(selected_tweets):
            formatted_tweets += f"{i+1}. {tweet}\n"

        # Create prompt using selected tweets and instruction
        prompt = self.args.prompt_template.format(
            tweets=formatted_tweets,
            instruction=self.args.instruction)
        llm_response_orig = self.get_response(prompt)
        llm_response = llm_response_orig

        if self.CLASS_NAMES[0] in llm_response:
            y_pred = self.CLASS_NAMES[0]
        elif self.CLASS_NAMES[1] in llm_response:
            y_pred = self.CLASS_NAMES[1]
        else:
            y_pred = random.choice(self.CLASS_NAMES)

        if self.CLASS_NAMES[0] in llm_response and self.CLASS_NAMES[1] in llm_response:  # noqa: E501
            y_pred = random.choice(self.CLASS_NAMES)

        if PRINT_LLM_INPUT_OUTPUT:
            print(f"\n======== LLM INFERENCE (author-{author_id}) ============================================================")  # noqa: E501
            print(f"====== Prompt:\n{prompt}")
            print(f"====== LLM Response:\n{llm_response_orig}")
            print(f"=====> Predicted Class: {y_pred} | True Class: {y_true}")
            print("===================================================================================\n")  # noqa: E501

        return y_pred
