from cdi_adapter.extract import evidence as E


def B(*t):
    return [{"text": x} for x in t]


def test_blocks_for_finds_the_lines_that_hold_the_words():
    blocks = B("Dr. Asha Rao", "Adv: CBC, LFT, FBS", "Tab Pantocid 40 1 tab OD", "Review after 2 weeks")
    assert E.blocks_for("CBC, LFT, FBS", blocks) == ["b2"]
    assert E.blocks_for("Review after 2 weeks", blocks) == ["b4"]
    assert E.blocks_for("something not on the page", blocks) == []


def test_a_line_split_over_blocks_is_found_by_its_pieces():
    blocks = B("Tab", "Pantocid 40", "x")
    assert "b2" in E.blocks_for("Tab Pantocid 40 OD", blocks)


def test_attach_fills_only_empty_evidence():
    blocks = B("Mrs Sumita Gupta", "Adv: TSH, FT4")
    p = {"patient": {"name": "Sumita Gupta"}, "investigations": [{"text": "TSH, FT4"}, {"text": "CBC", "evidence": ["b9"]}]}
    n = E.attach(p, blocks)
    assert p["patient"]["evidence"] == ["b1"] and p["investigations"][0]["evidence"] == ["b2"] and p["investigations"][1]["evidence"] == ["b9"] and n == 2
