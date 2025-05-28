import base64
import numpy as np
from typing import Dict
import random

import asyncio
import logging
import os, json
from typing import Any
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random
from time import sleep

import aiolimiter

import openai
from openai import AsyncOpenAI, OpenAIError
from anthropic import AsyncAnthropic

def aopenai_client(
):
    from dotenv import load_dotenv
    load_dotenv("./.env")
    client = AsyncOpenAI()
    AsyncOpenAI.api_key = os.environ["OPENAI_API_KEY"]
    return client

async def _throttled_openai_chat_completion_acreate_single(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    json_format: bool = False,
    n: int = 1,
):
    async with limiter:
        for _ in range(10):
            try:
                if json_format:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        response_format={"type": "json_object"},
                    )
                else:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
            except openai.RateLimitError as e:
                print("Rate limit exceeded, retrying...")
                sleep(random.randint(10, 20))  # 增加重试等待时间
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                sleep(random.randint(5, 10))
        return None

async def generate_from_openai_chat_completion_single(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
    json_format: bool = False,
    n: int = 1,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate_single(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            json_format=json_format,
            n=n,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if n == 1:
            if json_format:
                outputs.append(json.loads(response.choices[0].message.content))
            else:
                outputs.append(response.choices[0].message.content)
        else:
            if json_format:
                outputs.append([json.loads(response.choices[i].message.content) for i in range(n)])
            else:
                outputs.append([response.choices[i].message.content for i in range(n)])
    return outputs

async def _throttled_openai_chat_completion_acreate(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    json_format: bool = False,
    n: int = 1,
):
    async with limiter:
        for _ in range(10):
            try:
                if json_format:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        n=n,
                        response_format={"type": "json_object"},
                    )
                else:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        n=n,
                    )
            except openai.RateLimitError as e:
                print("Rate limit exceeded, retrying...")
                sleep(random.randint(10, 20))  # 增加重试等待时间
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                sleep(random.randint(5, 10))
        return None

async def generate_from_openai_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 150,
    json_format: bool = False,
    n: int = 1,
):
    # if "mini" in engine_name:
    #     requests_per_minute = 400
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            json_format=json_format,
            n=n,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if n == 1:
            if json_format:
                outputs.append(json.loads(response.choices[0].message.content))
            else:
                outputs.append(response.choices[0].message.content)
        else:
            if json_format:
                outputs.append([json.loads(response.choices[i].message.content) for i in range(n)])
            else:
                outputs.append([response.choices[i].message.content for i in range(n)])
    return outputs

async def _throttled_claude_chat_completion_acreate(
    client: AsyncAnthropic,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        try:
            return await client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        except:
            return None

async def generate_from_claude_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 150,
    n: int = 1,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    
    n_messages = []
    for message in messages:
        for _ in range(n):
            n_messages.append(message)

    async_responses = [
        _throttled_claude_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in n_messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    if n == 1:
        for response in responses:
            outputs.append(response.content[0].text)
    else:
        idx = 0
        for response in responses:
            if idx % n == 0:
                outputs.append([])
            idx += 1
            outputs[-1].append(response.content[0].text)

    return outputs

async def _throttled_litellm_acreate(
        model, 
        messages, 
        limiter: aiolimiter.AsyncLimiter, 
        retries=10
    ):
    # import pdb; pdb.set_trace()
    async with limiter:
        for _ in range(retries):
            try:
                response = await acompletion(model=model, messages=messages)
                return response['choices'][0]['message']['content']  # 获取返回的文本内容
            except Exception as e:
                # print(f"Error: {e}")
                # print(response)
                print(response)
                import pdb; pdb.set_trace()
                await asyncio.sleep(random.randint(5, 10))  # 异步睡眠，避免阻塞

        return "Error after retries"

async def gemini_litellm_chat_completion(
    model,
    messages,
    generation_config, 
    safety_settings,
    n: int = 1
):
    # if "flash" in model:
    #     requests_per_minute = 120
    # elif "pro" in model:
    #     requests_per_minute = 60

    # delay = 60.0 / requests_per_minute
    # limiter = aiolimiter.AsyncLimiter(1, delay)

    # n_messages = messages * n
    
    # # 异步请求
    # async_responses = [
    #     _throttled_litellm_acreate(
    #         model=model, 
    #         messages=message, 
    #         limiter=limiter
    #     )
    #     for message in n_messages
    # ]
    
    # # 收集所有响应
    # responses = await asyncio.gather(*async_responses)
    
    # # 如果n > 1，按批次组织响应
    # outputs = []
    # errors = 0
    # if n == 1:
    #     for response in responses:
    #         if "Error" in response:
    #             errors += 1
    #         outputs.append(response)
    # else:
    #     idx = 0
    #     for response in responses:
    #         if idx % n == 0:
    #             outputs.append([])
    #         idx += 1
    #         if "Error" in response:
    #             errors += 1
    #         outputs[-1].append(response)

    # print(f"Error number: {errors}")
    responses = await acompletion(model=model, messages=messages[0])
    outputs = responses['choices'][0]['message']['content'] 
    return outputs


async def _throttled_gemini_chat_completion_acreate(
    model,
    messages,
    generation_config,
    safety_settings,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(10):
            try:
                return await model.generate_content_async(
                    messages,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            except Exception as e:
                print(e)
                sleep(random.randint(5, 10))

        return None

async def generate_from_gemini_chat_completion(
    model,
    messages,
    generation_config,
    safety_settings,
    requests_per_minute: int = 500,
    n: int = 1,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    if "flash" in model.model_name:
        requests_per_minute = 400
    elif "pro" in model.model_name:
        requests_per_minute = 100

    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    
    n_messages = []
    for message in messages:
        for _ in range(n):
            n_messages.append(message)

    async_responses = [
        _throttled_gemini_chat_completion_acreate(
            model=model,
            messages=message,
            generation_config=generation_config,
            safety_settings=safety_settings,
            limiter=limiter,
        )
        for message in n_messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    errors = 0
    #import pdb; pdb.set_trace()
    if n == 1:
        for response in responses:
            try:
                response.resolve()
                outputs.append(response.text)
            except:
                errors += 1
                outputs.append("Error")
    else:
        idx = 0
        for response in responses:
            if idx % n == 0:
                outputs.append([])
            idx += 1
            response.resolve()
            outputs[-1].append(response.text)

    print(f"Error number: {errors}")
    return outputs
