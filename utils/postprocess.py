import re

def filter_postprocess(
    outputs: list[str],
    default: bool = False
) -> list[int]:
    def extract_label(text: str):
        text = text.strip().split('\n')[0]
        match = re.search(r'\b[01]\b', text)
        if match:
            return int(match.group())
        else:
            print(f'Cannot extract label from output: {text}')
            return int(default)
    
    results = [extract_label(output) for output in outputs]
    return results

def topk_join_postprocess(
    outputs: list[str]
):
    results = []
    for output in outputs:
        first_line = output.split('\n')[0].strip()
        match = re.search(r'\d+', first_line)
        if match:
            results.append(int(match.group()))
        else:
            print(f'WARNING: Can not extract llm output: {output}')
            results.append(0)
    return results