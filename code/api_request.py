from openai import OpenAI
import time


def completion(model, messages, max_try=3, prt=False):
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key='sk-i7hr97aad1rZEWEaC875D23a433d41458eD997498920FcA8'
        # api_key='sk-cDJxrPQJSE16fpKWnDAtWyr9KUR3Tl4uRq8xufBgYBHcN1no'
    )
    # client = OpenAI(
    #     base_url='https://api.gptoai.top/v1',
    #     api_key='sk-auk9uhNrfFzlRuLIJjccVgm1SYascQ6E6kZwzWDKmC9r8H6B'
    # )
    message = ''
    for i in range(max_try):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            message = response.choices[0].message.content
            if prt:
                print('{} Response:'.format(model))
                print(message)
            break
        except Exception as e:
            print(e)
            time.sleep(0.5)
            continue
    return message