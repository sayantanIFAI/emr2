from cdi_adapter.extract import age_sex as A


def blocks():
    return [{"text": "NN.", "bbox": [56, 242, 118, 266]}, {"text": "luwita.", "bbox": [144, 241, 253, 276]}, {"text": "q?u?ta.", "bbox": [291, 252, 389, 293]},
            {"text": "Gango?adhyay.", "bbox": [451, 257, 682, 321]}, {"text": "741f.", "bbox": [873, 272, 961, 314]},
            {"text": "Painless cure for your tooth pain", "bbox": [676, 199, 979, 215]}]


def test_the_age_sex_piece_is_found_in_the_name_row():
    assert A.find_token(blocks(), "Sumita Gupta Gangopadhyay") == (873, 272, 961, 314)
    assert A.find_token(blocks(), "Someone Else Entirely") is None


def test_a_value_needs_two_agreeing_readings():
    reads = [{"age": "74", "sex": "F"}, {"age": "74 yrs", "sex": "F"}, {"age": "71", "sex": "M"}]
    assert A.vote(reads) == ("74 yrs", "F")
    assert A.vote([{"age": "74", "sex": "F"}, {"age": "71", "sex": "M"}, {"age": None, "sex": None}]) == (None, None)


def test_apply_replaces_the_whole_page_answer_only_when_the_line_is_clear():
    class C:
        def vlm_json_ex(self, png, prompt, schema, **k):
            return {"age": "74", "sex": "F"}, None

    import cv2
    import numpy as np
    ok, png = cv2.imencode(".png", np.full((400, 1000, 3), 255, np.uint8))
    p = {"patient": {"name": "Sumita Gupta Gangopadhyay", "age_text": "30", "sex": "M"}}
    found = A.apply(C(), png.tobytes(), blocks(), p)
    assert p["patient"]["age_text"] == "74 yrs" and p["patient"]["sex"] == "Female" and found["sex_whole_page_said"] == "M"
