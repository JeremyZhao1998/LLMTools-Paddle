import re
import time

import openai


client = openai.OpenAI(api_key=<your/api/key>, base_url=<your/api/url>)


def translator(question, complete=True):
    translator_match = re.search(r'MT\((.*)\)', question)
    if translator_match:
        content = translator_match.group(1)
        content = content[: content.find(")")].strip()
        content_sent = ("Translate the following text into English "
                        "(directly return the translated text): ") + content
        retry_cnt, result = 0, None
        while retry_cnt < 5:
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful translator"},
                        {"role": "user", "content": content_sent},
                    ],
                    temperature=0.5,
                    max_tokens=512,
                    stream=False
                )
                result = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(e)
                print("Retrying...")
                retry_cnt += 1
                time.sleep(1)
        if result:
            if complete:
                return question[: question.find(']')] + ' -> ' + result + ' ]'
            else:
                return result
        return question
    else:
        return question


if __name__ == '__main__':
    print(translator("[MT(哈尔的移动城堡)]"))
