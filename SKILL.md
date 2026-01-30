# SKILL.md – Agentic Medical Device Reviewer（FDA / TFDA 審查官輔助）

> 本系統主要協助 **FDA 510(k)** 與 **臺灣 TFDA 第二、三等級醫療器材查驗登記** 之審查官與顧問，
> 透過多代理 (agents) + 多模型 (OpenAI / Gemini / Anthropic / Grok) 的方式，加速文件整理、審查規劃與紀錄撰寫。

---

## 1. 系統總覽

### 1.1 主要特色

- **WOW UI**  
  - Light / Dark 主題切換  
  - 英文 / 繁體中文介面  
  - 20 種「大師畫風」背景（Van Gogh、Monet、Picasso…），可用「Jackpot!」隨機抽選  
- **多模型路由**  
  - OpenAI：`gpt-4o-mini`, `gpt-4.1-mini`  
  - Google Gemini：`gemini-2.5-flash`, `gemini-2.5-flash-lite`  
  - Anthropic：`claude-3-5-sonnet-2024-10`, `claude-3-5-haiku-20241022`  
  - Grok (xAI)：`grok-4-fast-reasoning`, `grok-3-mini`  
- **API Key 管理**  
  - 若環境變數已設定，sidebar 只顯示「from environment」提示，不顯示明文 key  
  - 若未設定，可在 sidebar 輸入各家 API key，僅存在本次 session 記憶體  
- **Agentic Workflow**  
  - 每個 agent 皆可：  
    - 選擇模型  
    - 調整 `max_tokens`（預設 12000，可到 120k）  
    - 修改 prompt  
    - 在 UI 編輯輸出內容，作為下一個 agent 的輸入  
- **Dashboard**：  
  - 顯示各 tab / agent / model 的呼叫歷史與估算 tokens  
  - 以 Altair 圖表呈現使用趨勢  

---

## 2. 頁面 (Tabs) 功能說明

### 2.1 Dashboard

- 顯示：
  - 總執行次數
  - 各 tab 的呼叫分佈
  - 各模型使用次數
  - Tokens 使用量時間序列
  - 最近活動表（前 25 次呼叫紀錄）

### 2.2 TW Premarket – TFDA 第二、三等級查驗登記預審

**目標對象：TFDA / 顧問 / 申請人**

#### Step 1 – 線上填寫申請書（草稿）

- 在畫面中輸入：
  - 案件基本資料（案件類型、產地、產品等級、有無類似品、替代條款）  
  - 醫療器材基本資訊（中文/英文名稱、適應症、型號/主要成分）  
  - 分類分級品項（主類別、A.xxxx 品項代碼與名稱）  
  - 醫療器材商資料（統一編號、名稱、地址、負責人、聯絡人等）  
  - 製造廠資訊（名稱、國別、地址、是否委託製造）  
  - 臨床前測試、品質管制、臨床證據等附件的「文字摘要」  
- 一鍵產生：**Markdown 版申請書草稿**（可下載/複製/再編輯）。

#### Step 2 – 輸入預審/形式審查指引

- 支援：
  - 上傳 PDF / TXT / MD 指引檔案  
  - 直接貼上指引文字  
- 系統整合成一段文字，供預審代理參考。

#### Step 3 – 形式審查 / 完整性檢核（Agent：`tw_screen_review_agent`）

- 使用者可：
  - 選擇模型（預設 `gemini-2.5-flash`）  
  - 調整 max_tokens 與 prompt  
- 輸入為：
  - 申請書 Markdown + 預審指引  
- 輸出為：
  - Markdown 預審報告（有表格 + 評語），並支援編輯與後續串接。

#### Step 4 – 申請書文字優化（Agent：`tw_app_doc_helper`）

- 將申請書 Markdown 作為輸入，進行：
  - 結構與標題層級優化  
  - 文句修飾  
  - 以「※待補：...」標註明顯不足之處  
- 適合用於：
  - 產出更乾淨版本再轉 PDF/Word  
  - 作為補件回覆草稿基礎（可自行調整 prompt 要求）

---

### 2.3 510(k) Tab – FDA 510(k) Intelligence

- 輸入：
  - Device name, 510(k) number, sponsor, product code, additional context  
- 預設 Agent：`fda_510k_intel_agent`  
- 輸出：
  - 長篇 510(k) 情資摘要，含多個 markdown 表格  
- 適用：
  - 審查前快速建立「該案/類似案」的背景認知  
  - 顧問側準備 pre-sub 或 510(k) package 時作內部分析草稿  

---

### 2.4 PDF → Markdown

- 上傳 PDF，指定起迄頁，先由 `pypdf` 擷取文字。  
- 使用 Agent：`pdf_to_markdown_agent` 將原始文字整理為乾淨 Markdown。  
- 適用：
  - 將舊版紙本審查紀錄/指南數位化  
  - 將申請書附件轉為可供 LLM 後續分析的格式  

---

### 2.5 510(k) Review Pipeline（簡化版）

- Step 1：提交資料 → 結構化 Markdown  
  - 將貼上的 510(k) 提交文件以 LLM 重組，分章節頁面。  
- Step 2：Checklist（手工或由 `guidance_to_checklist_converter` 產出）  
- Step 3：Review Memo  
  - 以結構化提交資料與 Checklist 為輸入，使用 `review_memo_builder`（或預設模型）產出審查報告草稿。  

---

### 2.6 Note Keeper & Magics

專為審查官的個人筆記設計。

#### Step 1 – Note → Structured Markdown

- Agent：`note_structurer_agent`  
- 將雜亂筆記整理為：
  - 產品與案件概述  
  - 風險/疑慮  
  - 測試與資料  
  - 待辦/補件  

#### Magic 1 – AI Formatting

- Agent：`note_ai_formatting_agent`  
- 不改變內容，僅優化標題層級、分段與條列。

#### Magic 2 – AI Keywords + 手動顏色標註

- LLM 版：`note_keywords_agent` 建議關鍵字與顏色。  
- UI 版：輸入關鍵字（逗號分隔）+ 選色，系統直接在 Markdown 中包上 `<span style="color:...">`。

#### Magic 3 – AI Summary

- Agent：`summary_entities_agent` or `note_summary` 使用預設模型。  
- 輸出：
  - 主管用摘要 bullet  
  - 一段整體情況概述  

#### Magic 4 – AI Action Items

- Agent：`note_action_items_agent`  
- 產出待辦/補件/內部會議等行動清單表格。

#### Magic 5 – AI Glossary & Entities

- Agents：`note_glossary_agent`, `note_entity_table_agent`, `note_gap_finder_agent`  
- 功能：  
  - 術語表（中英文對照、說明）  
  - 20+ 個關鍵實體表（實體、類型、上下文、審查相關性）  
  - 法規與資料缺口分析  

---

### 2.7 Agents Config

- 以表格方式展示目前 `agents.yaml` 中所有代理。  
- 提供：
  - 原始 YAML 編輯器  
  - 上傳/下載 agents.yaml  
- 允許你在 UI 中調整：
  - name / category / description  
  - model / temperature / max_tokens  
  - system_prompt / user_prompt_template  

---

## 3. Agents 能力總覽（31 個）

> 下列為 `agents.yaml` 中的主要代理，按主題分類。

### 3.1 FDA 510(k) 與通用文件

1. **fda_510k_intel_agent**  
   - 510(k) 情資彙總與多表格摘要。

2. **k510_summary_structurer_agent**  
   - 將 510(k) Summary 重整為標準章節 Markdown。

3. **k510_entity_extractor_agent**  
   - 抽取 device name, K number, product code, predicates, tests, standards 等欄位。

4. **k510_diff_agent**  
   - 新舊版本比較，特別針對適應症與技術、測試變更。

5. **summary_entities_agent**  
   - 對大型文件做長篇摘要 + 多欄位實體表格。

6. **risk_register_agent**  
   - 風險與行動登錄表 (Risk Register)。

7. **benefit_risk_agent**  
   - 利益-風險摘要段落（供內部 memo 使用）。

8. **standard_mapping_agent**  
   - 測試項目 ↔ 標準 (IEC/ISO 等) 對照表。

9. **software_cybersecurity_agent**  
   - 軟體確效與網路安全資料整理與完整性檢視。

10. **emc_safety_agent**  
    - 電氣安全與 EMC 測試資料整理與缺漏檢視。

11. **biocompatibility_agent**  
    - 生物相容性資料依 ISO 10993 架構表示與簡要評估。

12. **labeling_translation_agent**  
    - 中英文標籤/說明書對照、翻譯與不一致標註。

13. **regulatory_strategy_agent**  
    - 高階全球註冊策略摘要（FDA / TFDA / EU MDR）。

14. **review_memo_builder**  
    - 將 checklist 與審查結果整合成正式審查報告草稿。

15. **guidance_to_checklist_converter**  
    - 將法規/技術指引轉成審查清單。

16. **dynamic_agent_generator**  
    - 讀取指引 + 現有 agents.yaml，產生新的專用代理 YAML 片段。

---

### 3.2 TFDA 第二、三等級查驗登記專用

17. **tw_screen_review_agent**  
    - 形式審查（預審）代理，產出預審報告與缺漏清單。

18. **tw_app_doc_helper**  
    - 協助撰寫與優化 TFDA 查驗登記申請書內容。

19. **tw_qms_checker_agent**  
    - QMS/QSD 證明文件與申請人/製造廠一致性檢查。

20. **tw_labeling_checker_agent**  
    - TFDA 標籤與中文說明書載明事項檢核。

21. **tw_preclinical_checker_agent**  
    - 臨床前測試 + 品質管制資料之完整性檢視。

22. **tw_clinical_evidence_agent**  
    - 臨床證據摘要與初步 sufficiency 評估。

---

### 3.3 Note Keeper & Magics

23. **note_structurer_agent**  
    - 將零散筆記整理成結構化 Markdown。

24. **note_ai_formatting_agent**  
    - 純版面與結構優化（不改變內容）。

25. **note_keywords_agent**  
    - 抽出關鍵字與建議著色。

26. **note_action_items_agent**  
    - 抽出所有待辦與補件行動項目。

27. **note_glossary_agent**  
    - 建立中英文術語表。

28. **note_entity_table_agent**  
    - 至少 20 個關鍵實體表格。

29. **note_gap_finder_agent**  
    - 從筆記中找出法規與資料缺口。

30. **pdf_to_markdown_agent**  
    - PDF 文字 → 清潔 Markdown 前處理。

31. **reviewer_chat_agent**  
    - 在單一案件上下文中提供 QA Chat，回答審查員問題並指引相關段落。

---

## 4. 推薦使用情境與鏈接 (Chaining) 範例

### 4.1 TFDA 新案預審

1. 在 **TW Premarket** tab 線上填寫申請書 → 產生 Markdown。  
2. 上傳/貼上 **預審指引**。  
3. 呼叫 `tw_screen_review_agent` → 預審報告。  
4. 若申請書文本需改善 → `tw_app_doc_helper`。  
5. 若同時要檢視 QMS/標籤/臨床前測試 → `tw_qms_checker_agent`, `tw_labeling_checker_agent`, `tw_preclinical_checker_agent`, `tw_clinical_evidence_agent`。  

### 4.2 FDA 510(k) 深度審查

1. 使用 `pdf_to_markdown_agent` 將 510(k) Summary 或技術文件轉 markdown。  
2. 使用 `k510_summary_structurer_agent` 重整為標準章節。  
3. 使用 `k510_entity_extractor_agent` 抽關鍵欄位與表格。  
4. 若有舊版本文件 → `k510_diff_agent` 比對。  
5. 根據對應指引 → `guidance_to_checklist_converter` 建立 checklist。  
6. 使用 `review_memo_builder` 建立審查報告草稿。  
7. 全程可用 `note_structurer_agent` / `note_action_items_agent` / `note_glossary_agent` 管理內部筆記。  

---

## 5. 實務建議（給審查官）

- **控制上下文長度**：  
  - 重要技術檔案建議先用 PDF → Markdown + 剪成幾個區塊再送入 agent。  
- **明確告知案件性質**：  
  - Prompt 中明說：「此案為 TFDA 第二等級 IVD」、「此案為 FDA 510(k) 傳統申請」有助於模型選擇正確語氣與重點。  
- **善用 Note Keeper**：  
  - 每個案件使用一份 Note，將不同 agent 的輸出片段貼入，最後再用 AI Summary / Risk Register 做收斂。  
- **避免過度依賴 LLM 作「實質決策」**：  
  - 模型適合作為「整理與提示」，真正的批准/不批准、補件要求仍需由審查官判斷。  
