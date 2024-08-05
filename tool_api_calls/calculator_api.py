import re


def calculator(api_call):
    # find Calculator(expression) in api_call
    expression_match = re.search(r'Calculator\((.*)\)', api_call)
    if expression_match:
        content = expression_match.group(1)
        if re.match(r'^\s*-?\s*\d+(\.\d+)?\s*%?\s*$', content):
            result = None
        else:
            try:
                result = str(eval(expression_match.group(1)))
            except Exception as e:
                result = None
    else:
        result = None
    if result is None:
        return api_call
    insert_idx = api_call.find(']')
    return api_call[:insert_idx] + ' -> ' + result + ' ]'
