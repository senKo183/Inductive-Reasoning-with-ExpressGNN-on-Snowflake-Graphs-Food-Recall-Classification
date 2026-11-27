"""
Tái tạo bộ dữ liệu kiểm thử trình đọc ExpressGNN (data/fda/parser_test)
và các bảng ánh xạ ID từ file gốc data/fda/food_recall.csv.
"""

from pathlib import Path
import pandas as pd

from data_process.preprocess import preprocess_kinship


# ============================================================
# TẦNG 1: Bộ từ khóa phân cấp (trích xuất từ EDA.ipynb)
# ============================================================

keyword_groups = {
    # BIOLOGICAL HAZARDS
    "Listeria": [
        "listeria monocytogenes",
        "l. monocytogenes",
        "listeria",
    ],
    "Salmonella": [
        "salmonella",
    ],
    "E_Coli": [
        "e. coli",
        "e.coli",
        "escherichia coli",
    ],
    "Cronobacter": [
        "cronobacter sakazakii",
        "cronobacter",
    ],
    "Cyclospora": [
        "cyclospora",
    ],
    "Hepatitis_A": [
        "hepatitis a",
    ],
    "Clostridium_Botulinum": [
        "clostridium botulinum",
        "botulinum",
    ],

    # ALLERGEN
    "Undeclared_Allergen": [
        "undeclared allergen",
        "undeclared milk",
        "undeclared peanut",
        "undeclared peanuts",
        "undeclared soy",
        "undeclared almond",
        "undeclared walnut",
        "undeclared egg",
        "undeclared tree nut",
        "undeclared",
        "allergen",
        "not declare",
        "not declared on the label",
        "not declared on label",
        "gluten free claim",
        "may contain gluten",
    ],

    # FOREIGN MATERIAL
    "Metal": [
        "metal shavings",
        "metal fragments",
        "metal fragment",
        "metal pieces",
        "metal piece",
        "metal wire",
        "metal",
    ],
    "Plastic": [
        "plastic pieces",
        "plastic piece",
        "plastic fragments",
        "plastic fragment",
        "plastic",
        "polyethylene",
        "resin",
    ],
    "Glass": [
        "glass fragments",
        "glass fragment",
        "glass pieces",
        "glass piece",
        "glass",
    ],
    "Rubber": [
        "rubber pieces",
        "rubber piece",
        "rubber fragments",
        "rubber",
        "food-grade rubber",
    ],
    "Wire": [
        "wire mesh",
        "flexible wire",
        "wire",
    ],
    "Cloth": [
        "cloth material",
        "cloth",
    ],
    "Foreign_Material": [
        "foreign material",
        "foreign object",
        "foreign matter",
    ],

    # CHEMICAL HAZARDS
    "Chloramphenicol": [
        "chloramphenicol",
    ],
    "Pesticide": [
        "pesticide residue",
        "pesticide",
        "violative pesticide",
    ],
    "Patulin": [
        "patulin",
    ],
    "Herbicide": [
        "herbicide",
        "haloxyfop",
    ],
    "Chemical_Contamination": [
        "chemical contamination",
        "contaminated with chemical",
        "incorrect chemical",
        "chemical added",
        "chemical",
    ],

    # PROCESSING
    "Under_Processing": [
        "under-processing",
        "under processing",
        "under-processed",
        "under processed",
    ],
    "Pasteurization": [
        "not adequately pasteurized",
        "not pasteurized",
        "not been pasteurized",
        "pasteurized",
        "pasteurization",
    ],
    "Temperature": [
        "not held at an appropriate temperature",
        "not held at proper temperature",
        "improper temperature",
        "inadequate temperature",
        "temperature abuse",
        "temperature",
    ],

    # SANITATION / FACILITY
    "Insanitary_Conditions": [
        "insanitary conditions",
        "unsanitary conditions",
        "insanitary",
    ],
    "Pest_Infestation": [
        "infestation",
        "mice",
        "mouse",
        "rodent",
        "pest",
    ],
    "Inadequate_Sanitizer": [
        "inadequate sanitizer",
        "sanitizer",
        "wash water",
    ],

    # GMP / MANUFACTURING
    "GMP_Violation": [
        "cgmp",
        "gmp",
        "good manufacturing practice",
    ],
    "Unlicensed_Facility": [
        "unlicensed facility",
        "unlicensed",
        "uninspected facility",
        "without a license",
    ],

    # LABELING
    "Labeling_Misbranding": [
        "misbranded",
        "misbranding",
        "labeling violation",
        "labeling",
        "incorrect label",
        "label",
    ],
    "Unapproved_Ingredient": [
        "unapproved ingredient",
        "unapproved drugs",
        "unapproved new drugs",
        "unapproved",
    ],

    # SPOILAGE / QUALITY
    "Spoilage": [
        "premature spoilage",
        "spoilage",
        "swelled containers",
        "foul odor",
    ],

    # ILLNESS / COMPLAINT
    "Illness_or_Complaint": [
        "consumer complaints",
        "consumer complaint",
        "reported illnesses",
        "reports of illnesses",
        "illnesses",
        "illness",
    ],
}


# ============================================================
# TẦNG 2: Hàm phân loại theo thứ tự ưu tiên mức độ nguy hiểm
# ============================================================

def classify_reason(reason):
    if pd.isna(reason):
        return "Missing"

    text = str(reason).lower().strip()

    biological = [
        "Listeria",
        "Salmonella",
        "E_Coli",
        "Cronobacter",
        "Cyclospora",
        "Hepatitis_A",
        "Clostridium_Botulinum",
    ]
    allergen = ["Undeclared_Allergen"]
    foreign_material = [
        "Metal",
        "Plastic",
        "Glass",
        "Rubber",
        "Wire",
        "Cloth",
        "Foreign_Material",
    ]
    chemical = [
        "Chloramphenicol",
        "Pesticide",
        "Patulin",
        "Herbicide",
        "Chemical_Contamination",
    ]
    processing = [
        "Under_Processing",
        "Pasteurization",
        "Temperature",
    ]
    sanitation = [
        "Insanitary_Conditions",
        "Pest_Infestation",
        "Inadequate_Sanitizer",
    ]
    manufacturing = [
        "GMP_Violation",
        "Unlicensed_Facility",
    ]
    labeling = [
        "Labeling_Misbranding",
        "Unapproved_Ingredient",
    ]
    quality = ["Spoilage"]
    illness = ["Illness_or_Complaint"]

    for category, keywords in {
        "Biological_Hazard": biological,
        "Undeclared_Allergen": allergen,
        "Foreign_Material": foreign_material,
        "Chemical_Hazard": chemical,
        "Processing_Violation": processing,
        "Sanitation_Violation": sanitation,
        "Manufacturing_Violation": manufacturing,
        "Labeling_Violation": labeling,
        "Quality_Issue": quality,
        "Illness_or_Complaint": illness,
    }.items():
        for key in keywords:
            for keyword in keyword_groups[key]:
                if keyword.lower() in text:
                    return category

    return "Other"


def build_parser_test(fda_dir: str = "data/fda", test_parse: bool = True):
    fda_path = Path(fda_dir)
    out_dir = fda_path / "parser_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đọc bảng dữ liệu gốc (29,471 dòng)
    df = pd.read_csv(fda_path / "food_recall.csv", dtype={"event_id": "string"})

    # 2. Chuẩn hóa ID thực thể (record_id, event_internal_id, firm_id)
    df["record_id"] = [f"record_{i:06d}" for i in range(1, len(df) + 1)]
    df["event_internal_id"] = "event_" + df["event_id"].fillna("unknown").astype(str)

    unique_firms = sorted(df["recalling_firm"].dropna().astype(str).unique())
    firm_map = {firm: f"firm_{idx:05d}" for idx, firm in enumerate(unique_firms, start=1)}
    df["firm_id"] = df["recalling_firm"].astype(str).map(firm_map)

    # Lưu 3 bảng ánh xạ ID
    pd.DataFrame({
        "record_id": df["record_id"],
        "source_csv_row": range(2, len(df) + 2),
        "recall_number": df["recall_number"],
        "event_id": df["event_id"],
        "event_internal_id": df["event_internal_id"],
    }).to_csv(fda_path / "record_id_mapping.csv", index=False)

    unique_events = sorted(df["event_id"].dropna().astype(str).unique())
    pd.DataFrame({
        "event_internal_id": [f"event_{e}" for e in unique_events],
        "event_id_original": unique_events,
    }).to_csv(fda_path / "event_id_mapping.csv", index=False)

    pd.DataFrame({
        "firm_id": [firm_map[f] for f in unique_firms],
        "recalling_firm_original": unique_firms,
    }).to_csv(fda_path / "firm_id_mapping.csv", index=False)

    # 3. Phân nhóm reason_category và chuẩn hóa classification, status
    df["reason_category"] = df["reason_for_recall"].apply(classify_reason)
    df["class_clean"] = df["classification"].fillna("Unknown").str.replace(" ", "_", regex=False)
    df["status_clean"] = df["status"].fillna("Unknown").str.replace(" ", "_", regex=False)

    # 4. Xuất 4 file chuẩn format ExpressGNN vào parser_test/
    facts_lines = []
    for row in df.itertuples(index=False):
        facts_lines.append(f"belongs_to_event({row.record_id}, {row.event_internal_id})")
        if pd.notna(row.firm_id):
            facts_lines.append(f"recalled_by({row.record_id}, {row.firm_id})")
        facts_lines.append(f"has_reason_category({row.record_id}, {row.reason_category})")
        facts_lines.append(f"has_classification({row.record_id}, {row.class_clean})")
        facts_lines.append(f"has_status({row.record_id}, {row.status_clean})")

    (out_dir / "facts.txt").write_text("\n".join(facts_lines) + "\n", encoding="utf-8")

    predicates_lines = [
        "belongs_to_event(recall_record, event)",
        "recalled_by(recall_record, firm)",
        "has_reason_category(recall_record, reason_category)",
        "has_classification(recall_record, classification)",
        "has_status(recall_record, status)",
        "reason_predicts_class(reason_category, classification)",
    ]
    (out_dir / "predicates.txt").write_text("\n".join(predicates_lines) + "\n", encoding="utf-8")
    (out_dir / "rules.txt").write_text("1.0 !belongs_to_event(x, e) v belongs_to_event(x, e)\n", encoding="utf-8")
    (out_dir / "queries.txt").write_text("!has_status(record_000001, Terminated)\n", encoding="utf-8")

    print(f"Đã tạo parser_test tại {out_dir} ({len(facts_lines)} facts).")

    # 5. Chạy thử preprocess_kinship của ExpressGNN để kiểm tra cú pháp
    if test_parse:
        fact_ls, rule_ls, query_ls = preprocess_kinship(
            str(out_dir / "predicates.txt"),
            str(out_dir / "facts.txt"),
            str(out_dir / "rules.txt"),
            str(out_dir / "queries.txt"),
        )
        print(f"preprocess_kinship() thành công: facts={len(fact_ls)}, rules={len(rule_ls)}, queries={len(query_ls)}")


if __name__ == "__main__":
    build_parser_test()

