from pathlib import Path

import munch
import numpy as np
import pytest
from astropy.io import fits
from matplotlib import pyplot as plt

from amical import load_bs_hdf5
from amical import loadc
from amical._cli.main import main

valid_commands = ["clean", "extract", "calibrate"]


@pytest.fixture()
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.mark.usefixtures("close_figures")
def test_clean(cli_datadir, tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "0")
    isz = 78
    ret = main(
        [
            "clean",
            "--datadir",
            str(cli_datadir),
            "--outdir",
            str(tmp_path),
            "--isz",
            str(isz),
        ]
    )

    input_file = sorted(cli_datadir.glob("*.fits"))
    with fits.open(input_file[0]) as hdu:
        data = hdu[0].data

    saved_file = list(Path(tmp_path).glob("*.fits"))
    with fits.open(saved_file[0]) as hdu:
        data_cleaned = hdu[0].data

    assert len(saved_file) == 1
    assert isinstance(data, np.ndarray)
    assert data_cleaned.shape[1] == isz
    assert ret == 0


@pytest.mark.usefixtures("close_figures")
@pytest.mark.parametrize("flag", ["--apod", "--sky", "--clip"])
def test_flag_clean(flag, cli_datadir, tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "0")
    isz = 78
    ret = main(
        [
            "clean",
            "--datadir",
            str(cli_datadir),
            "--outdir",
            str(tmp_path),
            "--isz",
            str(isz),
            flag,
        ]
    )
    assert plt.gcf().number == 3
    assert ret == 0


@pytest.mark.usefixtures("close_figures")
def test_clean_check(cli_datadir, tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "0")
    ret = main(
        ["clean", "--datadir", str(cli_datadir), "--outdir", str(tmp_path), "--check"]
    )
    assert ret == 0


@pytest.mark.usefixtures("close_figures")
def test_extract(cli_datadir, tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "0")

    isz = 78
    ret = main(
        [
            "clean",
            "--datadir",
            str(cli_datadir),
            "--outdir",
            str(tmp_path),
            "--isz",
            str(isz),
        ]
    )
    main(["extract", "--datadir", str(tmp_path), "--outdir", str(tmp_path)])

    output_file = sorted(Path(tmp_path).glob("*.h5"))

    bs = load_bs_hdf5(output_file[0])
    bs_keys = list(bs.keys())

    true_value_v2 = 0.61
    true_value_cp = 0.01

    assert len(output_file) == 1
    assert isinstance(bs, munch.Munch)
    assert len(bs_keys) == 13
    assert bs.vis2[0] == pytest.approx(true_value_v2, 1e-1)
    assert bs.cp[0] == pytest.approx(true_value_cp, 1e-1)
    assert ret == 0


@pytest.mark.usefixtures("close_figures")
def test_calibrate(cli_datadir, tmp_path, monkeypatch):
    for i in range(2):
        monkeypatch.setattr("builtins.input", lambda _: str(i))

        isz = 78
        ret = main(
            [
                "clean",
                "--datadir",
                str(cli_datadir),
                "--outdir",
                str(tmp_path),
                "--isz",
                str(isz),
            ]
        )
        plt.close("all")

    for i in range(2):
        monkeypatch.setattr("builtins.input", lambda _: str(i))
        main(["extract", "--datadir", str(tmp_path), "--outdir", str(tmp_path)])
        plt.close("all")

    responses = iter(["1", "0"])
    monkeypatch.setattr("builtins.input", lambda msg: next(responses))

    main(["calibrate", "--datadir", str(tmp_path), "--outdir", str(tmp_path)])

    output_file = sorted(Path(tmp_path).glob("*calibrated.fits"))

    cal = loadc(output_file[0])
    cal_keys = list(cal.keys())

    true_value_vis2 = 0.98479018
    true_value_wl = 4.286e-06

    assert len(output_file) == 1
    assert len(cal_keys) == 20
    assert cal.vis2[0] == pytest.approx(true_value_vis2, 1e-3)
    assert cal.wl[0] == pytest.approx(true_value_wl, 1e-9)
    assert ret == 0


@pytest.mark.usefixtures("close_figures")
@pytest.mark.parametrize("method", ["fft", "gauss", "unique", "square"])
def test_calibrate_method(method, cli_datadir, tmp_path, monkeypatch):
    for i in range(2):
        monkeypatch.setattr("builtins.input", lambda _: str(i))

        isz = 78
        ret = main(
            [
                "clean",
                "--datadir",
                str(cli_datadir),
                "--outdir",
                str(tmp_path),
                "--isz",
                str(isz),
            ]
        )
        assert ret == 0
        plt.close("all")

    for i in range(2):
        monkeypatch.setattr("builtins.input", lambda _: str(i))
        ret = main(
            [
                "extract",
                "--datadir",
                str(tmp_path),
                "--outdir",
                str(tmp_path),
                "--peakmethod",
                method,
            ]
        )
        assert ret == 0
        plt.close("all")

    responses = iter(["1", "0"])
    monkeypatch.setattr("builtins.input", lambda msg: next(responses))

    main(["calibrate", "--datadir", str(tmp_path), "--outdir", str(tmp_path)])

    output_file = sorted(Path(tmp_path).glob("*calibrated.fits"))

    cal = loadc(output_file[0])
    cal_keys = list(cal.keys())

    true_value_vis2 = 0.98479018
    true_value_wl = 4.286e-06

    assert len(output_file) == 1
    assert len(cal_keys) == 20
    assert cal.vis2[0] == pytest.approx(true_value_vis2, 1e-3)
    assert cal.wl[0] == pytest.approx(true_value_wl, 1e-9)
