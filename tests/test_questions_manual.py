import re
import sys

import requests

API_URL = "http://127.0.0.1:8000/ask"

QUESTIONS = [
    "打刻漏れをした場合、いつまでに修正申請が必要ですか？",
    "有給休暇は何日前までに申請する必要がありますか？",
    "経費精算の締め日はいつですか？",
    "在宅勤務はどのような条件で可能ですか？",
    "遅刻した場合の連絡方法は何ですか？",
    "交通費はどこまで経費として認められますか？",
    "有給休暇の繰り越しは可能ですか？",
    "勤怠修正の承認者は誰ですか？",
    "経費精算にレシートは必須ですか？",
    "欠勤する場合の申請方法を教えてください。",
    "残業代の計算方法は？",
    "健康診断の受診義務はありますか？",
    "社内研修の参加は必須ですか？",
    "退職金の計算方法は？",
    "育児休暇の取得条件は？",
    "社内メールの使用ルールは？",
    "社用車の利用方法は？",
    "福利厚生の詳細は？",
    "昇進の基準は？",
    "社内イベントの参加は？",
    "業務委託のルールは？",
    "情報セキュリティのポリシーは？",
    "社内報の購読は？",
    "社内システムのパスワード変更は？",
    "社内カフェテリアの利用は？",
    "社内スポーツクラブの参加は？",
    "社内図書館の利用は？",
    "社内相談窓口は？",
    "社内表彰制度は？",
    "社内ボランティア活動は？",
]


def has_chunk_id(answer: str) -> bool:
    """
    回答文に [chunk数字] の形で chunk_id が含まれているか
    """
    return bool(re.search(r"\[chunk\d+\]", answer))


def main():
    print("=== RAG QA Auto Test ===")
    ok_count = 0

    for i, q in enumerate(QUESTIONS, 1):
        resp = requests.post(API_URL, json={"question": q, "top_k": 6, "debug": False})

        if resp.status_code != 200:
            print(f"[{i}] ERROR: HTTP {resp.status_code}")
            continue

        data = resp.json()
        answer = data.get("answer", "")

        if has_chunk_id(answer):
            print(f"[{i}] PASS")
            ok_count += 1
        else:
            print(f"[{i}] FAIL")
            print("  Q:", q)
            print("  A:", answer)

    print(f"\nResult: {ok_count}/{len(QUESTIONS)} passed")

    if ok_count != len(QUESTIONS):
        sys.exit(1)


if __name__ == "__main__":
    main()
