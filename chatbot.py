from openai import OpenAI

client = OpenAI(api_key="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")  # Replace with your key

def get_chatbot_response(expression):
    prompt = f"The detected facial expression is: {expression}. Reply in a friendly and positive way."

    completion = client.chat.completions.create(
        model="gpt-4o",  # or gpt-4o-mini / gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "You are a friendly English chatbot. Give a positive, human-like response based on facial expression."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
