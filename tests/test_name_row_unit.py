from cdi_adapter.extract import resolve_llm as R


def b(x0, x1, y0=100, y1=130, text=""):
    return {"bbox": [x0, y0, x1, y1], "text": text}


def test_the_whole_name_row_is_joined_from_its_pieces():
    row = [b(40, 90, text="Mrs."), b(110, 230, text="Sumita"), b(250, 330, text="Gupta"), b(350, 620, text="Gangopadhyay")]
    assert R.name_row_box(row[3], row) == (40, 100, 620, 130)            # the last word alone was the old crop


def test_a_far_label_and_another_row_are_not_pulled_in():
    pieces = [b(40, 300, text="Mr Onkar Chowdhury"), b(900, 960, text="Age"), b(40, 300, y0=200, y1=230, text="next line")]
    assert R.name_row_box(pieces[0], pieces) == (40, 100, 300, 130)
