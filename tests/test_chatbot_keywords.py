from chatbot import ChatBot


def test_where_question_resolves_location():
    bot = ChatBot()
    answer_plain = bot.get_response("Myanmar")
    answer_where = bot.get_response("Where is Myanmar?")

    assert isinstance(answer_plain, str)
    assert isinstance(answer_where, str)
    # Heuristic: the "where" question should not be a completely unrelated answer
    # and should not be empty.
    assert answer_where.strip()
    # In many training sets, "where" responses contain location-style words
    keywords = ["country", "located", "asia", "southeast", "region"]
    if any(k in answer_plain.lower() for k in keywords):
        # If the base answer looks geographic, the "where" version should be similar
        for k in keywords:
            if k in answer_plain.lower():
                assert k in answer_where.lower()
                break

