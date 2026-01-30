# Antigravity Agentic Data+Guidance Studio — SKILL.md（更新版）

本系統是一個「代理式（Agentic）」工作空間，用於建立、整併、審閱與維護以下檔案：
- `defaultdataset.json`：資料集（TW case sets / 510(k) checklist sets）
- `defaultguide.md`：指引（BEGIN_SECTION 格式）
- `agents.yaml`：代理配置（可擴充與標準化）
- `SKILL.md`：共享知識（本文件，可被自動注入至所有 Agent 的 system prompt）

---

## 1) 系統契約（Contracts）

### 1.1 defaultdataset.json（標準 schema）
必須是 JSON，且至少包含：
```json
{
  "tw_cases": {
    "dataset_id": {
      "title": "string",
      "cases": [ { "any_fields": "allowed" } ]
    }
  },
  "k510_checklists": {
    "checklist_id": {
      "title": "string",
      "items": [
        { "section": "string", "item": "string", "expected": "string", "notes": "string" }
      ]
    }
  },
  "meta": {
    "generated_at": "ISO-8601",
    "generated_by": { "model": "string", "prompt": "string" }
  }
}
```

**重要規則**
- `tw_cases` / `k510_checklists` 必須是 object（dict）
- checklist items 必須包含 `section/item/expected/notes` 四個字串欄位
- 不可捏造官方要求或引用：若缺證據，寫 `TBD/需確認`

### 1.2 defaultguide.md（標準 section 格式）
必須包含一個或多個：
```md
<!-- BEGIN_SECTION: tw_xxx | TITLE: 標題 -->
...markdown body...
<!-- END_SECTION -->
```

或：

```md
<!-- BEGIN_SECTION: k510_xxx | TITLE: 標題 -->
...markdown body...
<!-- END_SECTION -->
```

**建議章節骨架（可選，但利於解析與一致）**
- `## 0. 審查目的`
- `## 1. 必要文件清單`
- `## 2. 關鍵欄位檢核`
- `## 3. 一致性檢核`
- `## 4. 常見缺失`
- `## 5. 建議輸出格式`

---

## 2) 可信度與安全原則（Non-hallucination Rules）

1. 不可冒充官方（FDA/TFDA）文件或宣稱「必須/硬性要求」，除非輸入明確提供。
2. 若缺引用或不確定，必須標示：
   - `TBD`
   - `需確認`
   - `未提供證據`
3. 對於法規/標準引用：
   - 若文本未給標準號/條款號，不可自行猜測。
   - 可提出「可能需要哪類來源」但必須標示 TBD。

---

## 3) 共享知識注入（SKILL Injection）

本系統可能將本文件（SKILL.md）自動注入到所有 Agent 的 system prompt 作為共享知識：
- 目的：讓各 Agent 使用一致的輸出格式、避免捏造、遵守 schema。
- 風險：SKILL 太長會增加 token 成本；必要時可在 UI 設定「最大注入字元數」。

---

## 4) 核心工作流程建議

### 4.1 Dataset+Guide Studio（上傳→標準化→編輯→下載）
- 上傳 `defaultdataset.json` / `defaultguide.md`
- 若非標準：
  - 先 deterministic 修補（若系統有）
  - 再用標準化代理轉換
- 編輯後下載

### 4.2 Mock Bundle Generator（指令生成）
輸入明確需求：
- TW datasets 數量、每組案例數
- 510(k) checklist items 數量
- defaultguide sections 數量（tw_ / k510_）

### 4.3 Multi-pack Combiner（多包合併）
- 上傳多份 dataset/guide
- 合併時避免覆蓋：若 id 重複，建議加後綴（例如 `__merge2`）

### 4.4 Guidance Ingestor（多份 guidance → bundle）
- 支援 PDF（建議先做頁碼擷取，必要時 OCR）
- 產出 defaultguide + mock dataset
- 對於 OCR 雜訊：允許在 `defaultguide.md` 內加入「來源清理註記」

---

## 5) FDA Guidance 產製工具輸出標準

### 5.1 Harmonization Mapper（固定欄位表格）
必須輸出表格（欄名不可更改）：
| Standard/Citation | Clause/Section | Guidance Section Ref | Evidence Expected | Status | Notes/Action |

Status 僅能用：
- Pass / Concern / Gap / TBD

### 5.2 Plain Language + Change Tracking
必須包含變更追蹤表：
| Original | New | Rationale |

### 5.3 Public Comment Analyzer（固定 JSON schema）
必須輸出 JSON：
```json
{
  "summary": {
    "themes": [{"theme":"...","count":1,"notes":"..."}],
    "top_risks": ["..."],
    "recommended_revisions": ["..."]
  },
  "items": [
    {
      "comment_id":"...",
      "theme":"...",
      "sentiment":"support|neutral|concern|oppose",
      "priority":"high|medium|low",
      "requested_change":"...",
      "suggested_response":"..."
    }
  ]
}
