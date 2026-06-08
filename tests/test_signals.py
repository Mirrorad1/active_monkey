from active_loop.signals import WorkerSignal, discretize, NUM_OBS, MODALITY_NAMES


def test_modalities_match_num_obs():
    assert MODALITY_NAMES == ["step_succeeded", "confidence", "error_count", "human_feedback"]
    assert NUM_OBS == [2, 3, 3, 3]


def test_discretize_success_high_conf_no_errors():
    sig = WorkerSignal(succeeded=True, confidence=0.9, error_count=0, human_feedback=None)
    obs = discretize(sig)
    assert obs == [1, 2, 0, 0]


def test_discretize_failure_low_conf_many_errors():
    sig = WorkerSignal(succeeded=False, confidence=0.1, error_count=7, human_feedback=None)
    assert discretize(sig) == [0, 0, 2, 0]


def test_discretize_human_feedback_encoded_when_present():
    sig = WorkerSignal(succeeded=True, confidence=0.5, error_count=1, human_feedback="right")
    assert discretize(sig) == [1, 1, 1, 2]
    sig_wrong = WorkerSignal(succeeded=True, confidence=0.5, error_count=1, human_feedback="wrong")
    assert discretize(sig_wrong)[3] == 1
