from cdi_adapter.extract import unread


def blk(text, y, state="single_engine", x0=100, x1=400):
    return {"text": text, "bbox": [x0, y, x1, y + 25], "recognition": {"state": state}}


def test_the_unreadable_review_line_at_the_foot_is_flagged():
    page = [blk("Rx", 100), blk("Tab Zedox 250 x 10d", 300), blk("Ponl:@metkto/uer", 700, "no_reading"), blk("? 2/3/24", 760),
            blk("PLEASE BRING THE PRESCRIPTION", 900, "printed")]
    got = unread.unreadable_order_lines(page)
    assert [t for _i, t in got] == ["Ponl:@metkto/uer"]
    assert unread.flags(page)[0]["code"] == "unreadable_order_line"


def test_a_readable_page_and_unreadable_lines_far_from_orders_raise_nothing():
    assert unread.unreadable_order_lines([blk("Tab X 1 tab", 100), blk("CBC, LFT", 200), blk("review 1 wk", 300)]) == []
    page = [blk("zzz", 100, "no_reading"), blk("a", 200), blk("b", 300), blk("c", 400), blk("d", 500), blk("e", 600), blk("f", 700)]
    assert unread.unreadable_order_lines(page) == []


def test_a_printed_line_is_never_flagged():
    assert unread.unreadable_order_lines([blk("Rx", 100), blk("garbled printed", 200, "printed"), blk("Adv", 250)]) == []


def test_a_phone_line_and_a_mostly_readable_line_are_not_flagged():
    page = [blk("Adv", 100), blk("Whatsapp : 9903220441 &9163803044", 200), blk("?1c-9100", 300), blk("review", 400)]
    assert unread.unreadable_order_lines(page) == []
