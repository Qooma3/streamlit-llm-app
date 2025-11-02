import os
import streamlit as st
from dotenv import load_dotenv

# .env から OPENAI_API_KEY を読み込み
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM（安定寄り）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 専門家プリセット
EXPERT_SYSTEMS = {
    "栄養士": (
        "あなたは有資格の管理栄養士です。エビデンスに配慮し、"
        "食品例・代替案・注意事項（アレルギー/服薬中の留意）も具体的に示してください。"
        "回答は日本語で、3点箇条書き＋一言まとめで簡潔に。"
    ),
    "フィットネストレーナー": (
        "あなたはパーソナルトレーナーです。安全を最優先し、"
        "フォーム要点・頻度・ボリューム・回復の考え方を提示してください。"
        "回答は日本語で、ウォームアップ→メイン→クールダウンの順で提案。"
    ),
    "経営コンサルタント": (
        "あなたは中小企業診断士相当のコンサルです。課題→原因→打ち手（短期/中期）→KPIの順に、"
        "実行可能なアクションと測定指標を提示してください。日本語で端的に。"
    ),
}

def run_llm(user_text: str, expert: str) -> str:
    """入力テキストと専門家選択を受け取り、LLMの回答を返す。"""
    system_msg = EXPERT_SYSTEMS.get(expert, "あなたは有能なアシスタントです。")
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("human", "{question}")]
    )
    chain = prompt | llm
    res = chain.invoke({"question": user_text})
    return res.content if hasattr(res, "content") else str(res)

st.title("Lesson21 提出：LLM機能つきWebアプリ（Streamlit × LangChain）")
st.caption("入力テキストをLLMへ投げ、選んだ『専門家の人格』で回答を返します。")

with st.expander("このアプリの概要 / 使い方", expanded=True):
    st.markdown(
        """
**概要**  
- 入力フォームに相談内容を記入し、専門家（ラジオボタン）を選んで「送信」。  
- LangChain経由でLLMにプロンプトを投げ、結果を画面に表示します。

**操作手順**  
1. 専門家の種類を選ぶ（栄養士 / フィットネストレーナー / 経営コンサルタント）  
2. 入力フォームに相談内容を記入  
3. 「送信」ボタンで回答を表示  

※ 注意：医療・法務などの判断は必ず専門家に確認してください。
        """
    )

expert = st.radio(
    "専門家の種類を選択してください：",
    list(EXPERT_SYSTEMS.keys()),
    horizontal=True,
)

user_text = st.text_area(
    "相談内容（プロンプト）を入力してください：",
    placeholder="例）コレステロールを下げたいので、1週間の食事と運動プランを提案してください。",
    height=140,
)

if st.button("送信", use_container_width=True):
    if not user_text.strip():
        st.warning("相談内容を入力してください。")
    else:
        with st.spinner("LLMに問い合わせ中..."):
            try:
                answer = run_llm(user_text, expert)
                st.success(f"▼ {expert}としての回答")
                st.markdown(answer)
            except Exception as e:
                st.error(f"エラーが発生しました：{e}")
