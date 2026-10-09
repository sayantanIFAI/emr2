from cdi_adapter.output import flat_table as F


def result(tests):
    return {
        "schema_version": "result.v1", "document_id": "11111111-1111-1111-1111-111111111111", "filename": "x.jpg", "source": {"channel": "webapp", "original_filename": "x.jpg"},
        "doc_type": "prescription", "is_handwritten": True, "page_count": 1, "status": "needs_check", "needs_check_count": 3,
        "quality": {"passed": True}, "flags": [], "provenance": {"prompt_version": "p1", "engines": {"vlm_served": "Qwen", "printed_ocr": "rapid"}},
        "intake": {"token_no": "T-17", "phone": "9830012345", "patient_name": "Sumita Gupta Gangopadhyay", "name_confirmed": False},
        "patient": {"name": {"value": "Sumita Gupta Gangopadhyay", "status": "needs_check"}, "age_text": {"value": "73 yrs", "status": "checked"},
                    "sex": {"value": "Female", "status": "checked"}},
        "doctor": {"name": {"value": "Dr. Debaditya Roy", "status": "checked"}, "department": {"value": "Rheumatology", "status": "checked"},
                   "clinic": {"name": {"value": "Apollo Clinic"}}},
        "organization": {"name": {"value": "Apollo Clinic", "status": "checked"}},
        "follow_up": {"text": "Review after 2 months", "kind": "interval", "interval_value": 2, "interval_unit": "months", "status": "needs_check"},
        "visits": [{"date": "2026-07-23", "is_latest": True}], "lab_tests": tests, "advice": [], "diagnoses": [], "medications": [],
    }


def test_one_row_per_recognised_test_with_the_prescription_columns_repeated():
    tests = [{"as_written": "CBC", "standard_name": "Complete blood count", "gate_recognised": True, "status": "needs_check"},
             {"as_written": "LFT", "standard_name": "Liver function test", "gate_recognised": True, "status": "needs_check"},
             {"as_written": "MRI LS spine", "status": "rejected"},                        # not a lab test: kept in raw_result only
             {"as_written": "Menz-e - 1", "gate_recognised": False, "status": "needs_check"}]   # not placed by the lists: not a row
    rows = F.flatten(result(tests))
    assert [r["lab_test_as_written"] for r in rows] == ["CBC", "LFT"] and [r["lab_test_seq"] for r in rows] == [1, 2]
    assert all(r["intake_token_no"] == "T-17" and r["patient_name"] == "Sumita Gupta Gangopadhyay" and r["patient_name_status"] == "needs_check" for r in rows)
    assert rows[0]["booking_needed"] == "yes" and rows[0]["booking_when_text"] == "after 2 months" and rows[0]["booking_with_doctor"] == "Dr. Debaditya Roy"
    assert rows[0]["doctor_clinic_name"] == "Apollo Clinic" and rows[0]["latest_visit_date"] == "2026-07-23"
    assert set(rows[0]) <= set(F.COLUMN_NAMES)


def test_a_prescription_without_tests_makes_one_row_marked_no_lab_test():
    rows = F.flatten(result([]))
    assert len(rows) == 1 and rows[0]["row_kind"] == "no_lab_test" and rows[0]["lab_test_seq"] is None


def test_the_ddl_names_every_column_once_and_the_search_columns_exist():
    assert len(F.COLUMN_NAMES) == len(set(F.COLUMN_NAMES))
    assert "CREATE TABLE IF NOT EXISTS prescription_flat" in F.DDL and "prescription_flat_token_phone" in F.DDL
    assert set(F.SCREEN_COLUMNS) <= set(F.COLUMN_NAMES)


def test_a_value_wrapped_in_a_status_object_is_unwrapped_for_a_boolean_column():
    r = result([])
    r["doctor"]["stamp_present"] = {"value": None, "status": "absent"}
    r["doctor"]["signature_present"] = {"value": True, "status": "checked"}
    row = F.flatten(r)[0]
    assert row["doctor_stamp_present"] is None and row["doctor_signature_present"] is True


def test_the_patient_phone_is_the_number_typed_at_upload_never_one_read_from_the_page():
    from cdi_adapter.output import json_connector as jc

    base = {"id": "d", "status": "validated", "original_filename": "x.jpg"}
    page_phone = {"patient": {"phone": "9354465808"}}                       # a letterhead's WhatsApp number the model read as the patient's
    typed = jc.build_result(jc.ResultInputs(document={**base, "phone": "9830012345"}, facts=[], blocks=[], pages=[], payload=page_phone))["patient"]["phone"]
    assert typed["value"] == "9830012345" and typed["status"] == "checked"
    none_typed = jc.build_result(jc.ResultInputs(document=base, facts=[], blocks=[], pages=[], payload=page_phone))["patient"]["phone"]
    assert none_typed["value"] is None and none_typed["status"] == "absent"
