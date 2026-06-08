from active_loop.alphabet import ALPHABET, V, normalize, encode, decode


def test_alphabet_is_28_chars():
    assert len(ALPHABET) == 28 and V == 28
    assert " " in ALPHABET and "." in ALPHABET and "a" in ALPHABET and "z" in ALPHABET


def test_normalize_lowercases_and_maps_unknown_to_space():
    assert normalize("Hello, World!") == "hello  world "
    assert all(c in ALPHABET for c in normalize("Tab\tand\n123"))


def test_encode_decode_round_trip():
    s = "the cat."
    assert decode(encode(s)) == s
    assert all(0 <= i < V for i in encode(s))
