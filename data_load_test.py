from data_loader.data_loader import SummaryDataLoader

# 데이터 디렉토리 설정
data_dir = "/Users/yujuyoung/Desktop/Merging/data/1.Training/train_labled/TL1/05.minute/2~3sent"

# 데이터 로더 생성
data_loader = SummaryDataLoader(data_dir=data_dir, batch_size=32)

# 데이터를 불러오기
try:
    inputs, labels, max_lens = data_loader.load_data()

    # 불러온 데이터 출력 (예시로 첫 5개의 데이터만 출력)
    print("----- Inputs (first 5) -----")
    for i in range(min(5, len(inputs))):
        print(f"{i+1}: {inputs[i]}")

    print("\n----- Labels (first 5) -----")
    for i in range(min(5, len(labels))):
        print(f"{i+1}: {labels[i]}")

    print("\n----- Max lengths (first 5) -----")
    for i in range(min(5, len(max_lens))):
        print(f"{i+1}: {max_lens[i]}")

except Exception as e:
    print(f"Error while loading data: {e}")
