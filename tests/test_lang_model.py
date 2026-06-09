import numpy as np
import pytest
from active_loop.lang_model import LangModel
from active_loop.alphabet import V, ALPHABET


TRAIN = ("the cat sat on the mat. the dog ran in the fog. "
         "the cat and the dog sat on the mat. ") * 6


def test_generate_returns_alphabet_text_of_length_n():
    lm = LangModel(seed=0)
    out = lm.generate("the ", n=20)
    assert isinstance(out, str) and len(out) == 20
    assert all(c in ALPHABET for c in out)


@pytest.mark.slow
def test_learning_changes_A_and_lowers_training_surprise():
    lm = LangModel(seed=0)
    before = lm.mean_surprise(TRAIN)
    lm.learn_stream(TRAIN, epochs=6)
    after = lm.mean_surprise(TRAIN)
    assert after < before


@pytest.mark.slow
def test_mean_surprise_beats_uniform_after_learning():
    lm = LangModel(seed=0)
    lm.learn_stream(TRAIN, epochs=6)
    assert lm.mean_surprise(TRAIN) < np.log(V)
