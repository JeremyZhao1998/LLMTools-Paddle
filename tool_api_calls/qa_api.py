import re
import time

import openai


client = openai.OpenAI(api_key=<your/api/key>, base_url=<your/api/url>)


def qa(question):
    qa_match = re.search(r'QA\((.*)\)', question)
    if qa_match:
        content = qa_match.group(1)
        content = content[: content.find(")")].strip()
        content_sent = "Answer this question briefly: " + content
        retry_cnt, result = 0, None
        while retry_cnt < 5:
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": content_sent},
                    ],
                    temperature=1.0,
                    max_tokens=128,
                    stream=False
                )
                result = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(e)
                print("Retrying...")
                retry_cnt += 1
                time.sleep(1)
        return question[: question.find(']')] + '-> ' + result + ' ]'
    else:
        return question
