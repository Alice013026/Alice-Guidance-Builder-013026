
# Antigravity Agentic Data+Guidance Studio — SKILL.md

本系統提供「資料集(defaultdataset.json) + 指引(defaultguide.md)」的全流程管理：
- 上傳、標準化、編輯、下載
- 多包合併（multi-pack combine）
- 由使用者指令生成 mock dataset + guidance
- 由多份 guidance（含 PDF）彙整成 defaultguide.md 並自動產生 mock defaultdataset.json
- agents.yaml / SKILL.md 管理與標準化

## 核心標準（Contracts）
### defaultdataset.json
必須是 JSON，並包含：
- tw_cases: object（多個 dataset）
- k510_checklists: object（多個 checklist）
- meta: object（可選，但建議）

### defaultguide.md
必須使用 section markers：
<!-- BEGIN_SECTION: tw_xxx | TITLE: ... -->
...markdown...
<!-- END_SECTION -->

或

<!-- BEGIN_SECTION: k510_xxx | TITLE: ... -->
...markdown...
<!-- END_SECTION -->

## 安全與真實性
- 系統輸出的 mock 內容必須標示為「合成/範例」，不可冒充官方文件。
- 若輸入來源不含明確要求，不可捏造法規或 FDA/TFDA 規範。
- 所有「缺漏/風險」應以「TBD/需確認/未提供證據」呈現。

## 建議工作流程
1) Dataset+Guide Studio：上傳既有 defaultdataset.json/defaultguide.md → 標準化 → 編輯
2) Generator：用 prompt 生成新的 mock datasets + guidance → 編輯 → 下載
3) Combiner：上傳多包 → 合併 → 編輯 → 下載
4) Guidance Ingestor：丟入多份 PDF/MD/TXT guidances → 產出 defaultguide.md + mock dataset → 編輯 → 下載
5) FDA Guidance Tools：outline → harmonization mapping → plain-language/FAQ

## Prompt 建議（可直接貼用）
### 生成 mock bundle
- 目標領域：
- 目標讀者：
- 法規/標準參考（若有）：
- 需要的 dataset 數量與每組案例筆數：
- defaultguide.md 要包含幾個 sections（tw_ / k510_）：
- 請輸出繁體中文。

### 從多份 guidance 萃取
- 請先列出每份 guidance 的主題與可能 section 切分
- 再輸出 defaultguide.md（含 BEGIN_SECTION）
- 再輸出 defaultdataset.json（mock datasets 與 checklist 與 guidance 對齊）
