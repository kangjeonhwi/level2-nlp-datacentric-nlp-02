import pandas as pd
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_llm_result(dataframe, prompt, output_path, slice_word, max_tokens=50, temperature=0.7, 
                   top_p=0.95, model_name='yanolja/EEVE-Korean-Instruct-10.8B-v1.0'):
    """
    LLM을 사용하여 데이터프레임의 텍스트를 처리하고 결과를 저장합니다.

    :param dataframe: 처리할 텍스트가 포함된 pandas DataFrame
    :param prompt: LLM에 전달할 프롬프트 템플릿
    :param output_path: 결과를 저장할 파일 경로
    :param slice_word: LLM 출력을 슬라이싱할 기준 단어
    :param max_tokens: 생성할 최대 토큰 수 (기본값: 50)
    :param temperature: 생성 다양성 조절 파라미터 (기본값: 0.7)
    :param top_p: 상위 확률 샘플링 파라미터 (기본값: 0.95)
    :param model_name: 사용할 LLM 모델 이름 (기본값: 'yanolja/EEVE-Korean-Instruct-10.8B-v1.0')
    :return: 처리된 텍스트가 포함된 새로운 DataFrame

    # 예시 사용법
    df = pd.read_csv("input_data.csv")
    output_path = "corrected_headlines.csv"
    get_llm_result(df, PROMPT_TEMPLATE, output_path, "교정:")
    """
    # 토크나이저와 모델 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(DEVICE)

    new_df = dataframe.copy()

    # tqdm을 사용하여 진행 상황 표시
    for tidx, text in tqdm(enumerate(dataframe['text']), total=len(dataframe)):
        formatted_prompt = prompt.format(input_text=text)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 생성된 텍스트에서 필요한 부분만 추출
        cleaned_text = generated_text.split(slice_word)[-1].strip()
        
        new_df.loc[tidx, 'text'] = cleaned_text

    # 결과를 CSV 파일로 저장
    new_df.to_csv(output_path, index=False)
    print(f"결과가 {output_path}에 저장되었습니다.")
    return new_df

# 프롬프트 템플릿 예시
PROMPT_TEMPLATE = """
당신은 전문적인 한국어 신문 헤드라인 교정 전문가입니다. 주어진 헤드라인의 오류를 최소한으로 수정하고, 원문의 의도와 핵심 내용을 최대한 유지해야 합니다. 다음 지침을 엄격히 따라주세요:

1. 원문 분석: 헤드라인의 주요 키워드와 전체적인 의미를 파악합니다.
2. 최소 수정: 명백한 오류만을 수정하고, 불필요한 변경은 피합니다.
3. 맥락 유지: 원문의 의도와 문맥을 최대한 유지합니다.
4. 오타 수정: 명확한 오타만 수정하고, 의심스러운 경우 원문을 유지합니다.
5. 문법 교정: 문법적 오류가 명확한 경우에만 수정합니다.
6. 불필요한 추가 금지: 원문에 없는 정보를 임의로 추가하지 않습니다.
7. 확신 없는 경우 유지: 교정이 확실하지 않은 경우, 원문을 그대로 유지합니다.

예시:
1. 원문: 문인 당2, 4동 민관2동 7사위 철거
교정: 문인동 2, 4동과 민관동 2동 7층 아파트 철거

2. 원문: WEF 일련의 이벤트에서 창조경제 논의
교정: WEF, 일련의 이벤트에서 창조경제 논의

3. 원문: 김종길 리먼 사태처럼 충격적 인 6Mq
교정: 김종길 "리먼 사태처럼 충격적인 상황 우려"

이제 다음 헤드라인을 위의 지침과 예시를 참고하여 최소한으로 교정해주세요. 확실하지 않은 부분은 원문 그대로 유지하세요:

원문: {input_text}
교정:
"""

def remove_outer_quotes(input_string):
    """
    문자열의 양 끝에 있는 따옴표를 제거하고 공백을 정리합니다.

    이 함수는 입력된 문자열의 시작과 끝에 있는 작은따옴표(')와 큰따옴표(")를 모두 제거합니다.
    또한 문자열의 앞뒤 공백도 제거합니다. 입력이 float 타입인 경우 공백 문자열을 반환합니다.

    Parameters:
    input_string (str or float): 처리할 입력 문자열 또는 float 값

    Returns:
    str: 따옴표와 앞뒤 공백이 제거된 문자열. 입력이 float인 경우 공백 문자열(' ') 반환.

    Examples:
    >>> remove_outer_quotes('"Hello, World!"')
    'Hello, World!'
    >>> remove_outer_quotes("'Python'")
    'Python'
    >>> remove_outer_quotes(' "  Spaces  " ')
    'Spaces'
    >>> remove_outer_quotes(3.14)
    ' '
    """
    if type(input_string) == float:
        return ' '
    # 문자열의 앞뒤 공백을 제거
    cleaned = input_string.strip()
    
    # 작은따옴표와 큰따옴표 확인
    while cleaned.startswith('"') or cleaned.startswith("'"):
        cleaned = cleaned[1:]
    while cleaned.endswith('"') or cleaned.endswith("'"):
        cleaned = cleaned[:-1]
    
    return cleaned


def merge_dataframes(df_list):
    """
    형식이 동일한 DataFrame들의 리스트를 받아 하나의 DataFrame으로 병합합니다.

    Parameters:
    df_list (list): 병합할 pandas DataFrame들의 리스트

    Returns:
    pd.DataFrame: 병합된 단일 DataFrame

    Raises:
    ValueError: 입력 리스트가 비어있거나 DataFrame들의 열이 서로 다른 경우 발생
    """
    # 빈 리스트 처리
    if not df_list:
        raise ValueError("입력 리스트가 비어 있습니다.")

    # DataFrame 개수가 1개인 경우 바로 반환
    if len(df_list) == 1:
        return df_list[0]

    # 모든 DataFrame의 열이 동일한지 확인
    first_df_columns = df_list[0].columns
    for df in df_list[1:]:
        if not df.columns.equals(first_df_columns):
            raise ValueError("모든 DataFrame의 열이 동일해야 합니다.")

    # DataFrame 병합
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df