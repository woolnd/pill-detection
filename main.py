from src.annotations import load_annotations
from src.make_yolo import *

# ---------- 실행 ----------
def main():
    """전체 변환 순서

    입력: 없음
    반환: 없음. data/yolo/ 를 새로 만들고 요약을 출력한다.

    동작:
        0. 이전 결과(data/yolo/)가 있으면 지운다.
        1. JSON 읽기 -> 이상한 이미지 빼기
        2. 클래스 번호표 만들기
        3. 조합 단위로 train/val 나누기
        4. train, val 파일 쓰기 + data.yaml 쓰기
        5. 요약 출력 (train 에 한 번도 안 나온 클래스도 확인)
    """
    # 0. 이전 결과가 남아 있으면 예전 분할과 섞이니까 지우고 새로 만든다
    if OUT.exists():
        shutil.rmtree(OUT)

    print("[1] 이상한 박스 걸러내기")
    ann = remove_bad_images(load_annotations())

    print("[2] 클래스 번호 만들기")
    class_ids, class_to_idx = make_class_map(ann)

    print("[3] train/val 나누기")
    train, val = split_by_combo(sorted(ann))  # sorted(ann) = 이미지 파일명 정렬 리스트

    print("[4] 파일 쓰기")
    write_split(ann, train, "train", class_to_idx)
    write_split(ann, val, "val", class_to_idx)
    write_yaml(class_ids)

    # 5. 확인: val 에만 있고 train 에 없는 클래스는 학습이 안 된다
    train_classes = set(b["class_id"] for n in train for b in ann[n])
    missing = sorted(set(class_ids) - train_classes)
    print(f"\n클래스 {len(class_ids)}개 / train {len(train)}장 / val {len(val)}장")
    print("train에 없는 클래스 (학습 안 됨):", missing)
    print("저장 위치:", OUT)


if __name__ == "__main__":
    main()
    