import re

def get_not_fit_pattern(text, pattern_list) : 
    """
    특정 패턴들에 해당되지 않는 문자들을 반환해주는 함수

    Parameters : 
    - text (str)
    - pattern_list (list) : 패턴들에 해당하는 리스트

    Returns : 
    - str : text의 문자들 중 pattern_list에 해당되지 않는 문자들
    """
    non_matching_chars = set(text)

    for pattern in pattern_list:
        matching_chars = set(re.findall(pattern, text))
        non_matching_chars -= matching_chars
    
    if not non_matching_chars :
        return ""
    return f"[{''.join(re.escape(char) for char in non_matching_chars)}]"


def remove_pattern_if_exceeds(data, pattern, max_num = 0):
    """
    데이터프레임 내의 텍스트에서 패턴의 개수가 지정된 값을 초과하는 경우,
    해당 패턴을 제거하는 함수

    Parameters :
    - data (pd.DataFrame)
    - pattern (str) : 정규식 패턴
                        - 자음+모음 한글 : r'[가-힣]'
                        - 특수문자 : r'[^\w\s]'
                        - 공백 : r'\s'
                        - 숫자 : r'\d+'
                        - 자음 : r'[ㄱ-ㅎ]'
                        - 모음 : r'[ㅏ-ㅣ]'
                        - 영어 : r'[a-zA-Z]'
                        - 한자 : r'[\u4e00-\u9fff]'
                        - 위에 해당되지 않는 것(others) : "others"
    - max_num (int) : 허용되는 pattern의 최대 개수.
                      pattern이 이 값을 초과하면 모든 pattern 제거
    
    Returns :
    - pd.DataFrame : 'text' 열의 pattern이 제거된 데이터프레임.
    """
    if pattern == 'others' :
        pattern_list = [
            r'[가-힣]',     # 한글 완성형 문자
            r'[^\w\s]',     # 특수문자
            r'\d+',         # 숫자
            r'\s',          # 공백 문자
            r'[ㄱ-ㅎㅏ-ㅣ]', # 한글 자음/모음
            r'[a-zA-Z]',    # 영어 알파벳
            r'[\u4e00-\u9fff]' # 한자
        ]
        
        data['text'] = [
            re.sub(get_not_fit_pattern(text, pattern_list), '', text) 
            if len(re.findall(get_not_fit_pattern(text, pattern_list), text)) > max_num else text
            for text in data['text']
        ]
    else :
        data['text'] = [
            re.sub(pattern, '', text) if len(re.findall(pattern, text)) > max_num else text
            for text in data['text']
            ]

    return data