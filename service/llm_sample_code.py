from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:60002/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model = "rinna/qwen2.5-bakeneko-32b-instruct-gptq-int4"

def generate_result(history):
    result = client.chat.completions.create(
            model=model,
            messages=history,
            # stream = True,
            temperature=0.7,
            top_p=0.8,
            max_tokens=512,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
    return result

history = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]
response = generate_result(history)
print(response.choices[0].message.content)
