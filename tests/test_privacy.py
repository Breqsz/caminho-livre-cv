import numpy as np

from caminho_livre_cv.privacy import blur_regions


def noise(h=100, w=100):
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_blur_changes_region_only():
    f = noise()
    out = blur_regions(f, [(10, 10, 50, 50)])
    assert not np.array_equal(out[10:50, 10:50], f[10:50, 10:50])
    assert np.array_equal(out[80, 80], f[80, 80])


def test_input_not_mutated():
    f = noise()
    before = f.copy()
    blur_regions(f, [(10, 10, 50, 50)])
    assert np.array_equal(f, before)


def test_box_partially_outside_is_clipped():
    f = noise()
    out = blur_regions(f, [(-20, -20, 30, 30), (90, 90, 200, 200)])
    assert out.shape == f.shape
    assert not np.array_equal(out[0:30, 0:30], f[0:30, 0:30])


def test_zero_area_box_ignored():
    f = noise()
    out = blur_regions(f, [(10, 10, 10, 40), (200, 200, 300, 300)])
    assert np.array_equal(out, f)


def test_even_ksize_ok():
    f = noise()
    out = blur_regions(f, [(10, 10, 50, 50)], ksize=50)
    assert out.shape == f.shape
