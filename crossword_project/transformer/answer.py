from transformers import pipeline

def main():
    # --- Extractive Question Answering --- #
    # using: https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad
    question_answerer = pipeline("question-answering",
                                model = "monologg/koelectra-base-v2-finetuned-korquad-384",
                                tokenizer = "monologg/koelectra-base-v2-finetuned-korquad-384")

    # text = r"""
    # 한국의 수도는 서울이지만, 영국의 수도는 런던이다.
    # """
    #
    # questions = [
    #     "한국의 수도는?",
    #     "영국의 수도는?"
    # ]

# https://www.apple.com/kr/ios/ios-15/features/ 맥락유지
    text = r"""
        잇따르는 요청 간의 맥락을 Siri가 전보다 더욱 잘 파악하게 되었습니다. 덕분에 방금 질문한 내용을 기반으로 Siri와 대화하듯 질문을 
        이어갈 수 있죠. 예를 들어 “서울 N타워 영업 시간이 언제까지야?”라고 물어본 다음 “거기까지 가는 데 얼마나 걸려?”라고 이어서 물어볼 
        수 있죠. 그럼 Siri가 두 질문이 연관된 질문이라는 사실을 이해한답니다.
    """
    questions = [
        "서울 N타워 영업 시간이 언제까지야?",
        "거기까지 가는 데 얼마나 걸려?"
    ]

    for qus in questions:
        result = question_answerer(**{"question": qus, "context": text})
        print(qus, "->", result)
    # print(...)


if __name__ == '__main__':
    main()
