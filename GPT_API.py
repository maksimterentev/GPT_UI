# Maksim Terentev
# GPT API
# Last changes: 04/06/2023
# Version 1.2.0

import openai

def conversation(model, system_content, user_content, max_tokens = 50, temperature = 0):
    if model == "text-davinci-003":
        response = openai.Completion.create(
            engine = model,
            prompt = system_content + user_content,
            max_tokens = max_tokens,
            temperature = temperature,
            logprobs = 3)
        return response.choices[0].text.strip()
    elif model == "gpt-3.5-turbo" or model == "gpt-4":
        response = openai.ChatCompletion.create(
            model = model,
            max_tokens = max_tokens,
            temperature = temperature,
            stop = None,
            messages = [
                #{"role": "system", "content": system_content}, 
                {"role": "user", "content": system_content + user_content}])
        return response['choices'][0]['message']['content'] 
    
    







