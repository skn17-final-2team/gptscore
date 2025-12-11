import os, sys
import json
# 절대 경로 지정 
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from final_runpod_server.sllm_model import build_agent, process_transcript_with_chunks
from final_runpod_server.main_model import load_model_q, load_faiss_db, escape_curly
from gptscore.log.gptscore_utils import Sample, compute_gptscore_for_sample
from gptscore.log.judge_score_utils import Sample as JudgeSample, compute_judge_score_for_sample


db_path = '/workspace/final_runpod_server/faiss_db_merged/'
vector_store, embedding_model = load_faiss_db(db_path)


# ===== 메인 실행부 =====
if __name__ == "__main__":
    user_domain = input("도메인 입력 (accounting, design, marketing_economy, it): ").strip()
    if user_domain.upper() == "ALL" or user_domain == "" :
        domain_filter = None
    else:
        domain_filter = user_domain

    # 모델 연결 (1.5b 파튜 기본값 설정됨)
    model = load_model_q()
    agent = build_agent(model=model, vector_store=vector_store, domain=domain_filter)

    while True:
        print("\n" + "="*60)
        print("회의록 전문을 입력하세요!")
        print("- 긴 전문은 자동으로 청크로 나눠서 처리됩니다")
        print("- 전체 전문 기반으로 안건/요약/태스크를 추출합니다")
        print("- 종료하려면 'exit' 입력")
        print("="*60 + "\n")

        query = input("전문: ")
        if query.lower() in ["exit", "quit"]:
            print("종료합니다.")
            break

        # 청크 처리 및 전체 요약/태스크 추출
        result = process_transcript_with_chunks(agent=agent, transcript=query, max_chunk_tokens=1500)

        # 결과 출력
        print("\n" + "="*60)
        print("최종 결과")
        print("="*60 + "\n")

        if result["chunk_results"]:
            print(f"✅ {len(result['chunk_results'])}개 청크 처리 완료\n")

        print("📝 안건/요약:")
        print("-" * 60)
        if isinstance(result["full_summary"], dict) and "error" in result["full_summary"]:
            print(f"❌ 에러: {result['full_summary']['error']}")
        else:
            print(result["full_summary"])

        print("\n📋 태스크:")
        print("-" * 60)
        if isinstance(result["full_tasks"], dict) and "error" in result["full_tasks"]:
            print(f"❌ 에러: {result['full_tasks']['error']}")
        else:
            print(result["full_tasks"])

        print("\n" + "="*60 + "\n")

        # JSON 형식으로도 출력 (최종 결과)
        try:
            result_json = json.dumps(result, ensure_ascii=False, indent=2)
            print("\n JSON 결과 :")
            print(result_json)
        except:
            pass

        # ========================
        default_user_request = (
            "이 회의록을 기반으로 요약, 이슈, 후속 태스크를 한국어로 정리하라."
        )

        # full_summary / full_tasks를 문자열로 정규화
        if isinstance(result["full_summary"], dict):
            summary_text = json.dumps(result["full_summary"], ensure_ascii = False, indent = 2)
        else:
            summary_text = str(result["full_summary"])

        if isinstance(result["full_tasks"], dict):
            tasks_text = json.dumps(result["full_tasks"], ensure_ascii=False, indent =2)
        else:
            tasks_text = str(result["full_tasks"])

        answer_text = (
            " [Summary] \n"
            + summary_text
            + "\n\n [Tasks] \n"
            + tasks_text
        )

        sample = Sample(
            transcript=query,
            user_request = default_user_request,
            answer = answer_text,
        )

        aspects = ["faithfulness", "instruction_following", "structure_clarity"]
        print("\nGPTScore 평가 결과:")
        for aspect in aspects:
            score = compute_gptscore_for_sample(sample, aspect)
            print(f"  - {aspect}: {score}")

        # =========================

        # ---------- LLM-as-a-judge (gpt-4o-mini) ----------
        judge_sample = JudgeSample(
            transcript=query,
            user_request=default_user_request,
            answer=answer_text,
        )

        # True → reasoning 포함, False → 점수만 (비용 절약)
        with_reasoning = True

        print("\nLLM-as-a-judge (gpt-4o-mini) 평가 결과 (1~5 점수 + 이유):")
        for aspect in aspects:
            score, reasoning = compute_judge_score_for_sample(
                judge_sample,
                aspect,
                with_reasoning=with_reasoning,
                return_reasoning=True,   # ← 여기 추가
            )
            print(f"[{aspect}] 점수: {score}")
            print("[이유]")
            print(reasoning)
            print("-" * 60)

