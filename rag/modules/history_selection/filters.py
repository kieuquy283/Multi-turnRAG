MEANINGLESS_PATTERNS = {

    "hello",
    "hi",
    "xin chào",
    "cảm ơn",
    "thank you",
    "ok",
    "oke",
    "bye"
}

def is_meaningful_turn(
    turn: dict
) -> bool:

    content = (
        turn.get("content", "")
        .strip()
        .lower()
    )

    if not content:
        return False

    if content in MEANINGLESS_PATTERNS:
        return False

    if len(content.split()) <= 2:
        return False

    return True