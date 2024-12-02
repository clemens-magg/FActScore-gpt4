from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.client = None
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100

        self.llama_model = None
        self.llama_tokenizer = None
        self.llama_pipeline = None
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        openai.api_key = api_key.strip()
        self.model = self.model_name
        self.client = OpenAI(api_key=openai.api_key)
        

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "gpt-4o-mini":
            response = self.call_GPT4(prompt, temp=self.temp)
            output = response.content
            return output, response
        elif self.model_name == "meta-llama-Llama-3.1-8B-Instruct":
            response = self.call_llama(prompt, temp=self.temp)
            output = response["choices"][0]["text"]
            return output, response
        else:
            raise NotImplementedError()


    def call_gpt4(self, prompt, model_name="gpt-4o-mini"):
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message
    
    
    def call_llama(self, prompt, model_name="meta-llama-Llama-3.1-8B-Instruct", temp=0.7):
        read_token = os.getenv("HF_TOKEN")
        self.load_llama(model_name, read_token, temp)

        response = self.llama_pipeline(prompt)[0]["generated_text"]

        return response
    

    def load_llama(self, model_name, token, temp=0.7, max_len=512):
        if self.llama_model is None or self.llama_tokenizer is None or self.llama_pipeline is None:
            self.llama_model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
            self.llama_tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            self.llama_pipeline = pipeline(
                "text-generation",
                model=self.llama_model,
                tokenizer=self.llama_tokenizer,
                max_new_tokens=max_len,
                framework="pt",
                batch_size=1,
                return_text=True,
                temperature=temp,
                top_p=0.9,
            )
