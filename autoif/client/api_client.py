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
    # async def create_chat_completions(self,
    #                         messages: Union[str, List[Dict[str, str]]],
    #                         model: str = None,
    #                         temperature: Optional[float] = 0.7,
    #                         top_p: Optional[float] = 1.0,
    #                         logprobs: Optional[bool] = False,
    #                         top_logprobs: Optional[int] = 0,
    #                         n: Optional[int] = 1,
    #                         max_tokens: Optional[int] = None,
    #                         stop: Optional[Union[str, List[str]]] = None,
    #                         stream: Optional[bool] = False,
    #                         presence_penalty: Optional[float] = 0.0,
    #                         frequency_penalty: Optional[float] = 0.0,
    #                         user: Optional[str] = None,
    #                         repetition_penalty: Optional[float] = 1.0,
    #                         session_id: Optional[int] = -1,
    #                         ignore_eos: Optional[bool] = False,
    #                         skip_special_tokens: Optional[bool] = True,
    #                         **kwargs):
    #     """Chat completion v1.

    #     Args:
    #         model: model name. Available from self.available_models.
    #         messages: string prompt or chat history in OpenAI format. Chat
    #             history example: `[{"role": "user", "content": "hi"}]`.
    #         temperature (float): to modulate the next token probability
    #         top_p (float): If set to float < 1, only the smallest set of most
    #             probable tokens with probabilities that add up to top_p or
    #             higher are kept for generation.
    #         n (int): How many chat completion choices to generate for each
    #             input message. Only support one here.
    #         stream: whether to stream the results or not. Default to false.
    #         max_tokens (int | None): output token nums. Default to None.
    #         stop (str | List[str] | None): To stop generating further
    #           tokens. Only accept stop words that's encoded to one token idex.
    #         repetition_penalty (float): The parameter for repetition penalty.
    #             1.0 means no penalty
    #         ignore_eos (bool): indicator for ignoring eos
    #         skip_special_tokens (bool): Whether or not to remove special tokens
    #             in the decoding. Default to be True.
    #         session_id (int): Deprecated.

    #     Yields:
    #         json objects in openai formats
    #     """
    #     assert stream is False, "Stream is not supported"
        
    #     if model is None:
    #         model = self.model
            
    #     pload = {
    #         k: v
    #         for k, v in locals().copy().items()
    #         if k[:2] != '__' and k not in ['self']
    #     }
        
    #     try:
    #         async with aiohttp.ClientSession() as session:
    #             async with session.post(self.chat_completions_v1_url,
    #                                   headers=self.headers,
    #                                   json=pload) as response:
    #                 content = await response.text()
    #                 output = json.loads(content)
    #                 return output
                        
    #     except asyncio.CancelledError:
    #         await session.close()
            
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
            
    

        