from typing import Optional, List, Union, Dict
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
import requests
import json
import asyncio
import aiohttp

class OpenAIClient:
    """Chatbot for LLaMA series models with turbomind as inference engine.

    Args:
        api_server_url (str): communicating address 'http://<ip>:<port>' of
            api_server
        api_key (str | None): api key. Default to None, which means no
            api key will be used.
    """

    def __init__(self,
                 base_url: str,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.models = asyncio.run(self.get_models())
        self.chat_completions_v1_url = f'{base_url}/chat/completions'
        self.headers = {'content-type': 'application/json'}
        if api_key is not None:
            self.headers['Authorization'] = f'Bearer {api_key}'
            
        if model is None:
            self.model = self.models[0]   
        else:
            assert model in self.models, f"Model {model} not found in {self.models}"
            self.model = model
            
    async def get_models(self):
        models = await self.client.models.list()
        assert models.data is not None, "No models found"
        return [model.id for model in models.data]
        
    def build_messages(self, inputs: str) -> List:
        messages = [{"role": "user",
                     "content": inputs}]
        return messages
            
    async def create_chat_completions(self, messages: List, n: int = 1, top_p: float = 1, temperature: float = 1, repetition_penalty: float = 1.0, frequency_penalty: float = 0.0, max_tokens: int = 2048) -> List[str]:
        assert isinstance(messages, list), "messages must be a list"
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            n=n,
            top_p=top_p,
            stream=False,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            extra_body={"repetition_penalty": repetition_penalty}
        )
        responses = [each.message.content for each in response.choices]
        return responses
            
    

        