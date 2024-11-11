import torch
from tqdm import tqdm
import re

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')

def paraphrase_headlines(input_path, output_path, prompt):
    """
    뉴스 헤드라인을 패러프레이징하는 함수

    :param input_path: 입력 CSV 파일 경로
    :param output_path: 출력 CSV 파일 경로
    :param prompt: LLM에 전달할 프롬프트
    """
    df = pd.read_csv(input_path)
    paraphrased_df = get_llm_result(df, prompt, output_path, "교정:")
    paraphrased_df.loc[:, 'text'] = paraphrased_df['text'].apply(remove_outer_quotes)
    paraphrased_df.to_csv(output_path, index=False)
    print(f"Paraphrased headlines saved to {output_path}")

if __name__ == '__main__':
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    input_path = os.path.join(DATA_DIR, "remove_symbols.csv")
    output_path = os.path.join(DATA_DIR, "paraphrased_with_llm.csv")

    prompt = """
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

    paraphrase_headlines(input_path, output_path, prompt)