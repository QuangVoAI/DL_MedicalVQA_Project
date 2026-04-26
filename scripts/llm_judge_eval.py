import json
import requests
import os
import time
from pathlib import Path
from tqdm import tqdm

import argparse

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH MẶC ĐỊNH
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="qwen2.5:14b")
  parser.add_argument("--input", type=str, default="data/merged_vqa_vi.json")
  parser.add_argument("--output", type=str, default="data/judge_results.json")
  return parser.parse_args()

args = parse_args()
MODEL_NAME = args.model
INPUT_CHECKPOINT = args.input
JUDGE_OUTPUT = args.output

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT DÀNH CHO BÁC SĨ GIÁM KHẢO (STRICT JUDGE)
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """Bạn là một Bác sĩ Chuyên khoa Thẩm định (Medical AI Auditor).
Nhiệm vụ của bạn là kiểm tra độ chính xác của bản dịch y khoa sau đây.

CÂU GỐC (TIẾNG ANH):
Question: {en_q}
Answer: {en_a}

BẢN DỊCH (TIẾNG VIỆT) CẦN KIỂM TRA:
Câu hỏi: {vi_q}
Câu trả lời: {vi_a}
Câu trả lời đầy đủ: {vi_full_a}

TIÊU CHÍ ĐÁNH GIÁ KHẮT KHE:
1. Độ chính xác Y khoa (0.5 điểm): Các thuật ngữ (phổi, tim, thùy, tràn dịch, gãy xương...) phải dịch đúng.
2. Độ trung thực (0.3 điểm): Không được bịa thêm thông tin không có trong bản gốc.
3. Ngữ pháp tự nhiên (0.2 điểm): Tiếng Việt phải trôi chảy, không lủng củng.

YÊU CẦU TRẢ VỀ:
- Nếu tổng điểm = 1.0 (Hoàn hảo): Trả về JSON với score: 1
- Nếu có bất kỳ lỗi nào (dù nhỏ): Trả về JSON với score: 0 và cung cấp bản sửa lỗi tốt nhất (fixed_vi_q, fixed_vi_a, fixed_vi_full_a).

TRẢ VỀ ĐỊNH DẠNG JSON DUY NHẤT:
{{
 "score": 1 hoặc 0,
 "reason": "Giải thích ngắn gọn lỗi nếu score=0",
 "fixed_vi_q": "Câu hỏi đã sửa (nếu cần)",
 "fixed_vi_a": "Câu trả lời đã sửa (nếu cần)",
 "fixed_vi_full_a": "Câu đầy đủ đã sửa (nếu cần)"
}}"""

# ─────────────────────────────────────────────────────────────────────────────
# HÀM GỌI OLLAMA
# ─────────────────────────────────────────────────────────────────────────────
def call_judge(en_q, en_a, vi_q, vi_a, vi_full_a):
  prompt = JUDGE_PROMPT.format(
    en_q=en_q, en_a=en_a, 
    vi_q=vi_q, vi_a=vi_a, vi_full_a=vi_full_a
  )
  
  payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": False,
    "format": "json",
    "options": {"temperature": 0.1} # Giảm nhiệt độ để kết quả ổn định nhất
  }
  
  try:
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    res = r.json().get("response", "{}")
    return json.loads(res)
  except Exception as e:
    return {"error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# LUỒNG CHÍNH
# ─────────────────────────────────────────────────────────────────────────────
def main():
  # 1. Load dữ liệu đầu vào
  if not os.path.exists(INPUT_CHECKPOINT):
    print(f"❌ Không tìm thấy file {INPUT_CHECKPOINT}")
    return
    
  with open(INPUT_CHECKPOINT, "r", encoding="utf-8") as f:
    data = json.load(f)
  
  # 2. Load tiến trình cũ (Resume) - Đảm bảo luôn là Dictionary
  judge_data = {}
  if os.path.exists(JUDGE_OUTPUT):
    try:
      with open(JUDGE_OUTPUT, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
        if isinstance(loaded_data, dict):
          judge_data = loaded_data
          print(f" Tiếp tục từ câu thứ {len(judge_data)}...")
        else:
          print("⚠️ File kết quả cũ không đúng định dạng (phải là dict), khởi tạo lại.")
    except Exception as e:
      print(f"⚠️ Lỗi khi load file cũ ({e}), khởi tạo lại.")

  # 3. Chạy Judge cho toàn bộ dataset
  if isinstance(data, list):
    items = list(enumerate(data))
  else:
    items = list(data.items())
  
  for rid, content in tqdm(items, desc="Đang thẩm định dữ liệu"):
    rid = str(rid) # Đảm bảo rid là string để so khớp với judge_data keys
    if rid in judge_data:
      continue # Bỏ qua câu đã chấm xong
      
    # Lấy thông tin cần chấm
    # Lưu ý: row gốc cần image_name, question... bạn có thể cần load dataset gốc nếu muốn đầy đủ EN
    # Ở đây mình giả định bạn đã có EN trong object hoặc chúng ta lấy từ checkpoint
    
    # Nếu trong checkpoint không có câu EN gốc, bạn cần merge nó vào trước. 
    # Giả định: bạn đang chạy script này ngay sau khi có kết quả dịch
    
    # Lấy thông tin cần chấm (hỗ trợ nhiều định dạng field)
    en_q = content.get("question") or content.get("en_q") or content.get("back_translation_en", "Unknown")
    en_a = content.get("answer") or content.get("en_a", "N/A")
    vi_q = content.get("question_vi", "")
    vi_a = content.get("answer_vi", "")
    vi_full_a = content.get("answer_full_vi") or vi_a # Dùng vi_a nếu không có full
    
    res = call_judge(
      en_q=en_q, 
      en_a=en_a,
      vi_q=vi_q,
      vi_a=vi_a,
      vi_full_a=vi_full_a
    )
    
    judge_data[rid] = {
      "original_data": content,
      "judge_feedback": res
    }
    
    # Lưu checkpoint sau mỗi 20 câu
    if len(judge_data) % 20 == 0:
      with open(JUDGE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(judge_data, f, ensure_ascii=False, indent=2)

  # 4. Lưu kết quả cuối cùng
  with open(JUDGE_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(judge_data, f, ensure_ascii=False, indent=2)
  
  print(f"✅ Đã thẩm định xong toàn bộ {len(judge_data)} mẫu!")
  print(f"Kết quả lưu tại: {JUDGE_OUTPUT}")

if __name__ == "__main__":
  main()
