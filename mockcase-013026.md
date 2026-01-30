你希望系統將 guidance 視為純文字 Markdown，還是拆成結構化欄位（目的/必備文件/一致性/常見缺失/輸出格式）以利自動生成 checklist 表？
你希望預設 guidance library 能否讓使用者「合併多份 guidance」並做差異比較（例如 Dataset #1 + 自訂追加規則）？
目前 case 的 spec_comp、tech_file_info 等是自由文字；你希望未來是否把它們拆成更結構化欄位（例如 JSON 子結構：規格表、材料表、測試列表）？
是否需要在 dataset 中加入「附件是否已備妥」的布林欄位（例如 has_cfs, has_qms_cert, has_risk_mgmt_summary），以便 completeness 更精準？
針對同一公司多案，你希望系統支援「公司資訊模板」套用（firm_name/addr/contact 共用），減少重複填寫嗎？
你希望匯入非標準 dataset 時，優先用規則映射（字典對照）還是優先用 LLM 標準化？對成本/準確性偏好如何？
若使用者上傳 dataset 含多列且某些列標準化失敗，你希望策略是：整批失敗、跳過失敗列、或保留並標記 standardize_error？
你希望 guidance 下載時除了 .md/.txt，是否也需要輸出成 .pdf（使用現有 reportlab）以利對外溝通？
除了 TW Premarket，你是否也希望為 510(k) pipeline 增加類似的「default checklist dataset + guidance」管理能力（載入/上傳/標準化/編輯/下載）以形成一致的資料管理體驗？
ファイルが選択されていませんファイルが選択されていません
Super, please create 3 mock case datasets in JSON and 3 respective review guidance in markdown. Ending with 20 comprehensive foollow up questions

Mock Case Dataset #1 (JSON) — 第二等級 / 輸入 / 一般醫材（周邊靜脈導管）
[
  {
    "doc_no": "衛授醫器字第1150003101號",
    "e_no": "MDE-2026-002410",
    "apply_date": "2026-07-05",
    "case_type": "一般申請案",
    "device_category": "一般醫材",
    "case_kind": "新案",
    "origin": "輸入",
    "product_class": "第二等級",
    "similar": "有",
    "replace_flag": "否",
    "prior_app_no": "",
    "name_zh": "一次性使用無菌周邊靜脈導管",
    "name_en": "Single-use Sterile Peripheral IV Catheter",
    "indications": "供醫療專業人員於周邊靜脈建立短期輸液或給藥通路使用。詳如核定之中文說明書。",
    "spec_comp": "導管材質：PU；規格：14G/16G/18G/20G/22G/24G；導管長度依規格；含不鏽鋼針芯與保護帽。",
    "main_cat": "J.一般醫院及個人使用裝置",
    "item_code": "J.1400",
    "item_name": "靜脈導管及其附件",
    "uniform_id": "27654321",
    "firm_name": "宏澤醫材貿易有限公司",
    "firm_addr": "臺北市中山區南京東路三段120號9樓",
    "resp_name": "林俊杰",
    "contact_name": "張雅雯",
    "contact_tel": "02-2711-8899",
    "contact_fax": "02-2711-8800",
    "contact_email": "reg@hongzemed.com.tw",
    "confirm_match": true,
    "cert_raps": false,
    "cert_ahwp": true,
    "cert_other": "輸液產品法規訓練證明（2025）",
    "manu_type": "單一製造廠",
    "manu_name": "VASCULAR ACCESS SOLUTIONS LTD.",
    "manu_country": "EU (Member State)",
    "manu_addr": "Industriestrasse 12, 60314 Frankfurt am Main, Germany",
    "manu_note": "原廠負責製造、EO 滅菌與最終放行；臺灣醫療器材商負責進口、倉儲與上市後監督。",
    "auth_applicable": "適用",
    "auth_desc": "附原廠授權登記書正本與中譯本；授權範圍涵蓋全部規格（14G–24G）及包裝型式。",
    "cfs_applicable": "適用",
    "cfs_desc": "提供出產國製售證明文件影本，含簽發機關、產品可辨識資訊與有效期。",
    "qms_applicable": "適用",
    "qms_desc": "製造廠 ISO 13485:2016 證書有效期至 2028-09；範圍涵蓋一次性血管通路產品之設計與製造。",
    "similar_info": "提供與已上市周邊靜脈導管比較：材質、導管規格、流量/壓降、針尖設計與安全護套差異；另附同類品許可證資訊。",
    "labeling_info": "中文標籤/說明書擬稿含：規格、無菌、一次性使用、滅菌方式（EO）、效期、批號、使用步驟、禁忌、注意事項。",
    "tech_file_info": "提供產品結構圖、材料清單、規格表、製程概要、滅菌確效摘要、包裝完整性與運輸模擬摘要。",
    "preclinical_info": "臨床前測試摘要：生物相容性（ISO 10993）、滅菌確效（ISO 11135）、包裝完整性/運輸模擬、導管抗折/抗拉、流量與漏液測試、針尖性能摘要。",
    "preclinical_replace": "",
    "clinical_just": "不適用",
    "clinical_info": "本品屬成熟技術之周邊靜脈導管；以臨床前性能、生物相容性、滅菌與風險管理支持安全有效，無臨床試驗需求。"
  },
  {
    "doc_no": "衛授醫器字第1150003102號",
    "e_no": "MDE-2026-002411",
    "apply_date": "2026-07-05",
    "case_type": "一般申請案",
    "device_category": "一般醫材",
    "case_kind": "新案",
    "origin": "輸入",
    "product_class": "第二等級",
    "similar": "有",
    "replace_flag": "否",
    "prior_app_no": "",
    "name_zh": "一次性使用無菌安全型周邊靜脈導管",
    "name_en": "Single-use Sterile Safety Peripheral IV Catheter",
    "indications": "供醫療專業人員建立周邊靜脈短期通路使用；具針刺防護機制以降低針刺風險。詳如核定之中文說明書。",
    "spec_comp": "規格：18G/20G/22G/24G；安全護套：自動回縮式；導管材質：PU；含固定翼（依型號）。",
    "main_cat": "J.一般醫院及個人使用裝置",
    "item_code": "J.1400",
    "item_name": "靜脈導管及其附件",
    "uniform_id": "27654321",
    "firm_name": "宏澤醫材貿易有限公司",
    "firm_addr": "臺北市中山區南京東路三段120號9樓",
    "resp_name": "林俊杰",
    "contact_name": "張雅雯",
    "contact_tel": "02-2711-8899",
    "contact_fax": "02-2711-8800",
    "contact_email": "reg@hongzemed.com.tw",
    "confirm_match": true,
    "cert_raps": false,
    "cert_ahwp": true,
    "cert_other": "輸液產品法規訓練證明（2025）",
    "manu_type": "單一製造廠",
    "manu_name": "VASCULAR ACCESS SOLUTIONS LTD.",
    "manu_country": "EU (Member State)",
    "manu_addr": "Industriestrasse 12, 60314 Frankfurt am Main, Germany",
    "manu_note": "安全機構為同廠設計製造；與一般型共用滅菌與放行流程（詳技術摘要）。",
    "auth_applicable": "適用",
    "auth_desc": "授權書涵蓋安全型系列與全部規格；附中譯本。",
    "cfs_applicable": "適用",
    "cfs_desc": "同系列產品出產國製售證明文件，含簽發機關資訊與效期。",
    "qms_applicable": "適用",
    "qms_desc": "ISO 13485:2016 證書有效期至 2028-09。",
    "similar_info": "提供安全型導管與市售同類品比較：安全機構設計、啟動條件、針刺防護覆蓋範圍。",
    "labeling_info": "標示擬稿補充：安全機構啟動說明、警語與廢棄處理注意事項。",
    "tech_file_info": "提供安全機構設計摘要、規格表、製程概要、滅菌確效與包裝完整性摘要。",
    "preclinical_info": "性能測試摘要：安全機構可靠度/啟動成功率、針尖防護覆蓋、流量/漏液、導管抗折；並附生物相容性、滅菌與包裝摘要。",
    "preclinical_replace": "",
    "clinical_just": "不適用",
    "clinical_info": "成熟導管產品，安全機構以功能/可靠度測試支持，無臨床試驗需求。"
  }
]
Review Guidance #1 (Markdown) — 周邊靜脈導管（第二等級/輸入）預審/形式審查指引
# 預審/形式審查指引（Mock）—第二等級輸入一般醫材：周邊靜脈導管（含安全型）

## 0. 審查目的
確認申請書與附件是否齊備、資訊是否一致（名稱/規格/製造廠/效期）、文件可追溯性是否足以進入技術審查。

---

## 1. 必要文件清單（預期應附）
1. 申請書（第二等級查驗登記申請書）
2. 醫療器材商許可執照（名稱/地址/負責人需一致）
3. 原廠授權登記書（輸入案通常必附；涵蓋全部規格與包裝型式）
4. 出產國製售證明（CFS 或同等效力文件；需可辨識產品與效期）
5. QMS/QSD 或 ISO 13485 證明（在效期內；範圍涵蓋設計/製造與產品類別）
6. 中文標籤/中文說明書/外盒標示擬稿（需含：無菌、一次性使用、滅菌方式、效期、批號）
7. 技術檔案摘要（結構、材料、規格表、適用範圍、製程簡述）
8. 滅菌確效摘要（EO：ISO 11135；需述 SAL、循環、放行方式）
9. 包裝完整性/運輸模擬摘要（如適用）
10. 生物相容性摘要（ISO 10993；對應接觸性質/時間）
11. 性能測試摘要（流量/漏液/抗折/抗拉等；安全型另需安全機構可靠度）
12. 風險管理摘要（ISO 14971；主要危害與控制）
13. 上市後監督/抱怨處理機制簡述（摘要）

---

## 2. 申請書關鍵欄位檢核
- 案件基本資料：案件類型/案件種類/產地/產品等級/替代條款
- 名稱一致性：中文/英文品名需與授權書、CFS、標示擬稿一致
- 規格完整性：Gauge、導管材質、長度、安全機構型式（若有）需可追溯
- 分類分級：主類別/品項代碼/品項名稱合理且完整
- 製造廠資訊：名稱/地址/國別需與 QMS/CFS 一致

---

## 3. 一致性與效期常見疑點
- 授權範圍是否漏列安全型系列或部分規格
- CFS 是否可辨識產品（系列名/型號）且仍在效期
- ISO 13485 範圍是否涵蓋「設計與製造」及相關產品
- 標示是否缺：批號/效期/滅菌方式/一次性使用警語
- 安全型是否提供安全機構啟動條件與可靠度證據摘要

---

## 4. 常見缺失（請列為缺漏/補件）
- 授權書未涵蓋全部規格/安全型
- CFS 過期或缺簽發機關/產品可辨識資訊
- EO 滅菌確效摘要未提 SAL 或放行方式
- 生物相容性無法對應接觸分類
- 安全機構缺可靠度/啟動成功率摘要
Mock Case Dataset #2 (JSON) — 第三等級 / 輸入 / IVD / 新案（多重呼吸道病毒 RT-PCR 檢測）
[
  {
    "doc_no": "衛授醫器字第1150004020號",
    "e_no": "IVD-2026-000730",
    "apply_date": "2026-08-18",
    "case_type": "一般申請案",
    "device_category": "體外診斷器材(IVD)",
    "case_kind": "新案",
    "origin": "輸入",
    "product_class": "第三等級",
    "similar": "有",
    "replace_flag": "否",
    "prior_app_no": "",
    "name_zh": "呼吸道病毒核酸檢測試劑（SARS-CoV-2/流感A/流感B）",
    "name_en": "Respiratory Virus RT-PCR Assay (SARS-CoV-2/Flu A/Flu B)",
    "indications": "用於體外定性檢測人類鼻咽拭子樣本中 SARS-CoV-2、Influenza A 與 Influenza B 病毒 RNA。供臨床診斷之輔助。詳如核定之中文說明書。",
    "spec_comp": "含抽提/擴增相關試劑、反應混合液、內控品（依套組）；適用指定即時 PCR 儀（詳技術檔案）。",
    "main_cat": "C.免疫學及微生物學",
    "item_code": "C.2450",
    "item_name": "核酸擴增檢驗系統",
    "uniform_id": "24550011",
    "firm_name": "維康診斷科技有限公司",
    "firm_addr": "桃園市龜山區復興一路168號6樓",
    "resp_name": "蔡承諭",
    "contact_name": "楊詩涵",
    "contact_tel": "03-397-1122",
    "contact_fax": "03-397-1133",
    "contact_email": "ivd@weicandiag.com.tw",
    "confirm_match": true,
    "cert_raps": false,
    "cert_ahwp": true,
    "cert_other": "IVD clinical performance training record (2025)",
    "manu_type": "單一製造廠",
    "manu_name": "GENOME DIAGNOSTICS GmbH",
    "manu_country": "EU (Member State)",
    "manu_addr": "BioPark 8, 69120 Heidelberg, Germany",
    "manu_note": "原廠負責設計、製造與最終放行；臺灣醫療器材商負責進口與在地法規/上市後監督。",
    "auth_applicable": "適用",
    "auth_desc": "附原廠授權登記書及中譯本；授權範圍涵蓋本套組與配套耗材（如適用）。",
    "cfs_applicable": "適用",
    "cfs_desc": "提供出產國製售證明或同等效力文件影本，含簽發機關、效期與產品可辨識資訊。",
    "qms_applicable": "適用",
    "qms_desc": "製造廠 ISO 13485:2016 證書有效期至 2027-11；範圍涵蓋 IVD 試劑設計與製造。",
    "similar_info": "提供與既有多重 RT-PCR 檢測之比較：標的基因區域、檢測通道配置、內控策略、相容儀器差異。",
    "labeling_info": "IFU/標籤擬稿含：用途、樣本型別、操作步驟、限制、結果判讀、品質控制、儲存條件、效期、批號。",
    "tech_file_info": "提供方法學摘要（引子探針設計原則）、性能規格（LoD、特異性）、相容儀器清單、干擾/交叉反應評估摘要與文件版本控管摘要。",
    "preclinical_info": "性能驗證摘要：LoD、精密度、再現性、交叉反應/干擾、臨床樣本或等同性資料摘要（依規範）；內控有效性與陰/陽性控制策略摘要。",
    "preclinical_replace": "",
    "clinical_just": "適用",
    "clinical_info": "提供臨床性能摘要：以多中心/回溯性樣本評估陽/陰性一致性、敏感度/特異度及分層分析（摘要）；並說明樣本收集、倫理與統計方法概述。"
  },
  {
    "doc_no": "衛授醫器字第1150004021號",
    "e_no": "IVD-2026-000731",
    "apply_date": "2026-08-18",
    "case_type": "一般申請案",
    "device_category": "體外診斷器材(IVD)",
    "case_kind": "新案",
    "origin": "輸入",
    "product_class": "第三等級",
    "similar": "有",
    "replace_flag": "否",
    "prior_app_no": "",
    "name_zh": "呼吸道病毒核酸萃取試劑（配套用）",
    "name_en": "Nucleic Acid Extraction Reagent (Companion Use)",
    "indications": "用於從人類呼吸道拭子樣本中萃取核酸，供相容之核酸檢測使用。詳如核定之中文說明書。",
    "spec_comp": "裂解液、洗滌液、洗脫液；適用指定自動化萃取平台（詳技術檔案）。",
    "main_cat": "C.免疫學及微生物學",
    "item_code": "C.2450",
    "item_name": "核酸擴增檢驗系統",
    "uniform_id": "24550011",
    "firm_name": "維康診斷科技有限公司",
    "firm_addr": "桃園市龜山區復興一路168號6樓",
    "resp_name": "蔡承諭",
    "contact_name": "楊詩涵",
    "contact_tel": "03-397-1122",
    "contact_fax": "03-397-1133",
    "contact_email": "ivd@weicandiag.com.tw",
    "confirm_match": true,
    "cert_raps": false,
    "cert_ahwp": true,
    "cert_other": "IVD clinical performance training record (2025)",
    "manu_type": "單一製造廠",
    "manu_name": "GENOME DIAGNOSTICS GmbH",
    "manu_country": "EU (Member State)",
    "manu_addr": "BioPark 8, 69120 Heidelberg, Germany",
    "manu_note": "作為配套萃取試劑；與檢測試劑共用品質系統與文件版本控管。",
    "auth_applicable": "適用",
    "auth_desc": "授權文件包含本萃取試劑與配套平台適用性說明；附中譯本。",
    "cfs_applicable": "適用",
    "cfs_desc": "同系列產品之製售證明文件（或同等效力文件）影本，含效期與簽發機關。",
    "qms_applicable": "適用",
    "qms_desc": "ISO 13485:2016 證書有效期至 2027-11。",
    "similar_info": "提供與同類核酸萃取試劑比較：平台相容性、萃取效率、抑制物去除策略。",
    "labeling_info": "IFU/標籤擬稿：樣本型別、操作步驟、儲存條件、效期、批號、注意事項。",
    "tech_file_info": "提供平台相容性摘要、萃取效率/抑制物評估摘要、文件版本控管摘要。",
    "preclinical_info": "性能摘要：萃取效率、抑制物影響、批間差與穩定性摘要；與配套 RT-PCR 系統之整體性能關聯說明（摘要）。",
    "preclinical_replace": "",
    "clinical_just": "不適用",
    "clinical_info": "萃取試劑以分析性能與系統整體性能支持，無獨立臨床試驗需求。"
  }
]
Review Guidance #2 (Markdown) — 多重呼吸道病毒 RT-PCR（第三等級/輸入 IVD）預審/形式審查指引
# 預審/形式審查指引（Mock）—第三等級輸入 IVD：多重呼吸道病毒 RT-PCR（含配套萃取）

## 0. 審查目的
確認 IVD 新案之文件齊備性、臨床/分析性能資料可追溯性、相容儀器/平台界定、以及文件版本控管是否足以進入技術審查。

---

## 1. 必要文件清單（預期應附）
1. 申請書（第三等級 IVD；產品分類分級與用途）
2. 醫療器材商許可執照（名稱/地址/負責人一致性）
3. 原廠授權登記書（輸入案通常必附；需涵蓋檢測試劑與配套萃取/耗材範圍）
4. 出產國製售證明（CFS 或同等效力文件；產品可辨識、效期、簽發機關）
5. QMS/QSD 或 ISO 13485（涵蓋 IVD 試劑設計/製造；在效期內）
6. IFU/標籤/外盒標示擬稿（用途、樣本型別、判讀、限制、QC、儲存/運輸、效期、批號）
7. 技術檔案摘要（方法學、標的基因、內控策略、相容儀器/平台、試劑組成）
8. 分析性能摘要（至少：LoD、精密度/再現性、交叉反應、干擾、攜帶污染/污染控制策略）
9. 臨床性能摘要（若適用）：研究設計、樣本數、比較方法、敏感度/特異度或一致性、統計摘要
10. 品質控制策略摘要（陰/陽性控制、內控、判讀失敗條件）
11. 風險管理摘要（ISO 14971；錯誤結果風險與控制）
12. 文件版本控管清單（IFU/Label/技術檔案版本、修訂日期與摘要）

---

## 2. 申請書關鍵欄位檢核
- 用途/適應症：定性/定量、用途敘述與臨床定位是否一致
- 樣本型別：鼻咽/咽拭子等是否明確且與性能資料一致
- 相容儀器/平台：PCR 儀與（若有）萃取平台是否界定清楚
- 產品組成：是否清楚區分檢測試劑、萃取試劑、控制品與耗材

---

## 3. 一致性與常見疑點
- CFS/授權/QMS 是否可辨識同一產品系列（名稱/型號一致）
- IFU 是否清楚描述判讀規則（多重標的的陽性/陰性/無效）
- LoD/交叉反應摘要是否與宣稱用途/樣本型別一致
- 文件版本控管是否清楚（避免 IFU 與性能摘要版本不一致）
- 配套萃取試劑是否有相容平台界定與性能摘要（至少分析性能）

---

## 4. 常見缺失（請列為缺漏/補件）
- 缺臨床性能摘要（或無法追溯研究設計/樣本數/比較方法）
- 相容儀器/平台界定不清（導致適用範圍不明）
- IFU 缺 QC/內控策略、結果判讀或限制
- 交叉反應/干擾摘要不足（未對應呼吸道常見病原或抑制物）
- 文件版本控管缺失（無版本號/修訂日期/修訂摘要）
Mock Case Dataset #3 (JSON) — 第二等級 / 國產 / 一般醫材（一次性電燒筆與配套電極）
[
  {
    "doc_no": "衛授醫器字第1150005150號",
    "e_no": "MDE-2026-003120",
    "apply_date": "2026-09-30",
    "case_type": "一般申請案",
    "device_category": "一般醫材",
    "case_kind": "新案",
    "origin": "國產",
    "product_class": "第二等級",
    "similar": "有",
    "replace_flag": "否",
    "prior_app_no": "",
    "name_zh": "一次性使用電燒筆（單極）",
    "name_en": "Single-use Electrosurgical Pencil (Monopolar)",
    "indications": "供醫療專業人員於手術中進行電燒切割與凝血用途，需搭配相容之高頻電燒主機使用。詳如核定之中文說明書。",
    "spec_comp": "一次性電燒筆含按鍵控制（CUT/COAG）；刀頭型式：刀片/針狀/球狀（依型號）；線長 3 m；相容端子規格詳技術檔案。",
    "main_cat": "I.一般及整形外科手術",
    "item_code": "I.4700",
    "item_name": "電外科手術器械及其附件",
    "uniform_id": "80334455",
    "firm_name": "鈞曜醫工股份有限公司",
    "firm_addr": "高雄市前鎮區新生路66號7樓",
    "resp_name": "葉家豪",
    "contact_name": "陳思妤",
    "contact_tel": "07-330-2211",
    "contact_fax": "07-330-2222",
    "contact_email": "ra@junyaomed.com.tw",
    "confirm_match": true,
    "cert_raps": false,
    "cert_ahwp": false,
    "cert_other": "IEC 60601-1/EMC basics training (2025)",
    "manu_type": "單一製造廠",
    "manu_name": "鈞曜醫工股份有限公司（前鎮廠）",
    "manu_country": "TAIWAN， ROC",
    "manu_addr": "高雄市前鎮區新生路66號5樓",
    "manu_note": "製程包含組裝、功能測試、絕緣/耐壓檢測與最終放行；滅菌方式：EO（委外滅菌並受控）。",
    "auth_applicable": "不適用",
    "auth_desc": "",
    "cfs_applicable": "不適用",
    "cfs_desc": "",
    "qms_applicable": "適用",
    "qms_desc": "ISO 13485:2016 證書有效期至 2028-04；範圍涵蓋外科一次性耗材之設計與製造。",
    "similar_info": "提供與市售電燒筆比較：刀頭型式、按鍵控制方式、端子相容性、線長與絕緣材料差異。",
    "labeling_info": "中文標籤/說明書擬稿含：相容主機/設定注意事項、一次性使用警語、滅菌方式、效期、批號、禁忌、使用與廢棄指引。",
    "tech_file_info": "提供端子相容性規格、材料清單（絕緣材料/導體）、結構圖、製程概述、委外滅菌受控摘要、包裝完整性摘要。",
    "preclinical_info": "測試摘要：電氣安全相關（如絕緣耐壓/漏電相關設計驗證，視適用性）、熱效應/凝血切割功能驗證摘要、端子連接可靠度、包裝完整性、滅菌確效與生物相容性（皮膚接觸材料）摘要。",
    "preclinical_replace": "",
    "clinical_just": "不適用",
    "clinical_info": "本品為成熟電外科附件；以性能與安全設計驗證、滅菌與風險管理支持，無臨床試驗需求。"
  },
  {
    "doc_no": "衛授醫器字第1150005151號",
    "e_no": "MDE-2026-003121",
    "apply_date": "2026-09-30",
    "case_type": "一般申請案",
    "device_category": "一般醫材",
    "case_kind": "新案",
    "origin": "國產",
    "product_class": "第二等級",
    "similar": "有",
    "replace_flag": "否",
    "prior_app_no": "",
    "name_zh": "一次性使用電外科電極（單極配套）",
    "name_en": "Single-use Electrosurgical Electrode Tips (Monopolar Accessories)",
    "indications": "供相容之電燒筆於手術中切割與凝血用途之電極耗材。詳如核定之中文說明書。",
    "spec_comp": "電極型式：刀片/針狀/球狀；材質：不鏽鋼；表面處理依型號；與相容電燒筆之裝配介面詳技術檔案。",
    "main_cat": "I.一般及整形外科手術",
    "item_code": "I.4700",
    "item_name": "電外科手術器械及其附件",
    "uniform_id": "80334455",
    "firm_name": "鈞曜醫工股份有限公司",
    "firm_addr": "高雄市前鎮區新生路66號7樓",
    "resp_name": "葉家豪",
    "contact_name": "陳思妤",
    "contact_tel": "07-330-2211",
    "contact_fax": "07-330-2222",
    "contact_email": "ra@junyaomed.com.tw",
    "confirm_match": true,
    "cert_raps": false,
    "cert_ahwp": false,
    "cert_other": "IEC 60601-1/EMC basics training (2025)",
    "manu_type": "單一製造廠",
    "manu_name": "鈞曜醫工股份有限公司（前鎮廠）",
    "manu_country": "TAIWAN， ROC",
    "manu_addr": "高雄市前鎮區新生路66號5樓",
    "manu_note": "電極加工與檢驗由同廠執行；EO 滅菌委外受控；與電燒筆共用文件版本控管。",
    "auth_applicable": "不適用",
    "auth_desc": "",
    "cfs_applicable": "不適用",
    "cfs_desc": "",
    "qms_applicable": "適用",
    "qms_desc": "ISO 13485:2016 證書有效期至 2028-04。",
    "similar_info": "提供電極與同類耗材比較：材質、形狀尺寸、表面處理與相容介面差異。",
    "labeling_info": "標示擬稿含：相容型號、一次性使用警語、滅菌方式、效期、批號與使用注意事項。",
    "tech_file_info": "提供尺寸規格與公差摘要、材料/表面處理摘要、裝配相容性摘要、包裝完整性與滅菌確效摘要。",
    "preclinical_info": "測試摘要：尺寸/表面品質檢驗、裝配牢固度、導電性/熱效應設計驗證摘要（依用途）、生物相容性（接觸材料）與滅菌/包裝摘要。",
    "preclinical_replace": "",
    "clinical_just": "不適用",
    "clinical_info": "成熟電外科附件，無臨床試驗需求。"
  }
]
Review Guidance #3 (Markdown) — 一次性電燒筆/電極（第二等級/國產）預審/形式審查指引
# 預審/形式審查指引（Mock）—第二等級國產一般醫材：一次性電燒筆與配套電極（單極）

## 0. 審查目的
確認耗材型電外科附件之：申請書欄位、相容性界定（主機/端子/型號）、滅菌與包裝、標示警語、以及基本性能/安全驗證摘要是否齊備可追溯。

---

## 1. 必要文件清單（預期應附）
1. 申請書（分類分級、用途、製造廠資訊）
2. 國產製造廠與醫療器材商資料一致性文件（登記資訊）
3. QMS/QSD 或 ISO 13485（在效期內；範圍涵蓋設計/製造）
4. 中文標籤/中文說明書/外盒標示擬稿
   - 一次性使用警語、相容主機注意事項、滅菌方式、效期、批號、禁忌/警告
5. 技術檔案摘要
   - 結構與材料、電氣連接/端子規格、刀頭/電極型式、使用情境
6. 相容性界定摘要
   - 相容主機品牌/型號或端子規格、使用模式（CUT/COAG）、限制條件
7. 滅菌確效摘要（EO：ISO 11135；SAL/放行方式/委外滅菌受控）
8. 包裝完整性/運輸模擬摘要（如適用）
9. 生物相容性摘要（接觸材料對應 ISO 10993）
10. 性能與安全設計驗證摘要
    - 連接可靠度、絕緣/耐壓設計驗證（依適用性）、功能性（切割/凝血）驗證摘要
11. 風險管理摘要（ISO 14971；熱傷害、誤用、相容性錯配等風險與控制）

---

## 2. 申請書關鍵欄位檢核
- 產品界定：電燒筆 vs 電極耗材的範圍、型號、差異是否清楚
- 相容性：是否明確說明相容主機/端子規格（避免適用範圍不明）
- 標示一致性：中文/英文品名與標示、技術摘要一致；批號/效期/滅菌方式齊備
- 製造與滅菌：委外滅菌受控說明是否提及（至少摘要）

---

## 3. 常見疑點與一致性檢核
- 相容性描述過於籠統（未界定端子或主機型號）
- IFU 未提示高頻手術風險或未提供使用限制（例如：不得在含可燃麻醉環境使用等，視產品聲明）
- 滅菌確效摘要未提 SAL/放行方式或委外受控不足
- 電氣/熱效應相關驗證摘要不足（僅描述但無驗證概述）

---

## 4. 常見缺失（請列為缺漏/補件）
- 缺 ISO 13485 或證書範圍不匹配
- 缺相容性界定（主機/端子/限制條件）
- 缺 EO 滅菌確效摘要（SAL、放行方式）
- 缺批號/效期/一次性使用警語於標示擬稿
- 缺風險管理摘要或未涵蓋熱傷害/相容性錯配等主要危害
20 Comprehensive Follow-up Questions
你希望每個「case dataset」包含幾筆案例最合適（1、2、5、10）來支援你預期的測試與 demo？
多案例 dataset 中，你偏好用什麼欄位做「案例列表顯示」的主標題（e_no、name_zh、apply_date、或自組合）？
你希望系統在載入 dataset 時是否要自動檢查並修正欄位值域（例如 product_class 限定第二/第三等級，origin 限定國產/輸入/陸輸）？
對於輸入案，是否要在 completeness/缺漏清單中加入「授權書/CFS/QMS 必備」的強制規則？若是，哪些情境可例外？
對 IVD 第三等級新案，你希望 guidance 是否把「臨床性能」視為形式審查必備（必附摘要），還是只要技術審查再補？
IVD 檢測若含「配套萃取」，你希望系統如何表達兩者關係：同一案內子產品、兩案綁定、或主案/配套案？你的審查流程偏好哪一種？
你希望 guidance 是否需要加入「文件版本控管最低要求」的具體欄位（版本號、修訂日期、修訂摘要、核准者）並要求在報告中逐項檢核？
對於含安全機構（如安全型導管），你希望形式審查是否要強制要求「可靠度/啟動成功率」摘要，或可僅在技術審查要求？
對於電外科附件，你希望 guidance 中是否要要求「相容主機清單」必附，或只要端子/介面規格界定即可？
是否需要在案例 schema 中額外加入 compatible_devices 或 compatibility_notes 類欄位，以避免相容性資訊只存在自由文字中？
你希望 CSV 匯出時，長文字欄位（如 preclinical_info）是否要做長度限制、換行保留、或另存成附檔欄位？
你希望 guidance 的下載是否要同時提供「審查代理版本」與「申請者填寫版本」（含待補欄位模板）？
你偏好 guidance 中的「必要文件清單」是否要採固定欄位（可機器解析）格式，例如每項含 doc_id、required、applies_if？
你希望系統是否要支援「依案例類型自動選擇 guidance」的規則（例如 IVD/輸入/第三等級 → 自動挑 Dataset #2 類模板）？
你希望 standardization（非標準 dataset → 標準 schema）時優先採用：規則映射、欄位相似度、還是 LLM？對成本與準確率取捨如何？
若使用者上傳 dataset 中混有中英欄位、不同日期格式、或布林值為「Y/N」，你希望系統採用哪些預設轉換規則？
是否需要在匯入時做「敏感資訊遮罩」選項（例如自動遮蔽電話/email/地址）以便在 demo 或分享時使用？
你希望 completeness 的分數是否要分層（基本欄位、附件摘要、文件一致性、風險管理）並各自給分，而不只單一百分比？
你希望「缺漏清單」是否要能一鍵匯出成 Markdown/CSV（作為補件追蹤表）？若要，欄位應包含哪些（缺漏項、嚴重度、建議補件、責任人、期限）？
這些 mock datasets/guidance 你希望更偏向哪個真實情境：TFDA 內部審查訓練、申請者自檢工具、或 AI 代理測試資料？不同情境會影響語氣與必備項嚴格程度，你偏好哪一種？
2m49s
46.7kt
9,877t
ファイルが選択されていません
