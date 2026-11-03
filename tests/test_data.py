import numpy as np
import pytest

from cruxtok.data.leakage import assert_no_leakage, build_blocklist, find_leaks, normalize
from cruxtok.data.text import fit_window


def test_fit_window_fills_with_eos():
    assert fit_window([5, 6], 5, 0) == [5, 6, 0, 0, 0]
    assert fit_window([5, 6, 7, 8, 9, 10], 4, 0) == [5, 6, 7, 8]


def test_no_eval_data_leakage():
    blocklist = build_blocklist(["A girl is styling her hair."])
    assert normalize("A  GIRL is styling her hair.") == "a girl is styling her hair."
    assert find_leaks(["  a girl IS styling   her hair. "], blocklist), "normalized exact match must hit"
    assert find_leaks(["Once upon a time. A girl is styling her hair. The end."], blocklist), "an embedded sentence must hit"
    assert not find_leaks(["A girl is brushing her hair."], blocklist)
    with pytest.raises(AssertionError, match="eval texts"):
        assert_no_leakage(["a girl is styling her hair."], blocklist)


def test_tinystories_windows_from_local_files(tmp_path):
    from cruxtok.data.tinystories import TinyStoriesWindows
    stream = np.array([5, 6, 0, 7, 8, 9, 10, 11, 0], dtype=np.uint16)
    stream.tofile(tmp_path / "train.bin"); stream.tofile(tmp_path / "valid.bin")
    np.save(tmp_path / "off.npy", np.array([0, 3, 9], dtype=np.int64))
    w = TinyStoriesWindows(4, 0, tmp_path / "train.bin", tmp_path / "off.npy", tmp_path / "valid.bin")
    assert w.window(0).tolist() == [5, 6, 0, 0] and w.window(1).tolist() == [7, 8, 9, 10]
    assert w.valid_windows(2).tolist() == [[5, 6, 0, 0], [7, 8, 9, 10]]
    first = w.sample(8, np.random.default_rng(0)); second = w.sample(8, np.random.default_rng(0))
    assert (first == second).all()


@pytest.mark.slow
def test_real_eval_sets_are_not_in_train_sources():
    blocklist = build_blocklist()
    assert len(blocklist) > 50000
    from cruxtok.data.xsum import load_summaries
    assert_no_leakage(load_summaries("train"), blocklist, "XSum train")
