"""
@author: Anthony Soulain (UGA - IPAG)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

In this example, we provide a way to clean and extract data coming from IFU
instruments like the VLT/SPHERE. The main idea is to deal with each spectral
channels individually and combine them afterward.
--------------------------------------------------------------------
"""
import os

import amical

# Set your own cleaning paramters (appropriate for SPHERE)
clean_param = {"isz": 149, "r1": 70, "dr": 2, "apod": True, "window": 65, "f_kernel": 3}

# Set the AMICAL parameters
params_ami = {
    "peakmethod": "fft",
    "bs_multi_tri": False,
    "maskname": "g7",
    "fw_splodge": 0.7,
    "filtname": "YJ",  # Use the appropriate filter (only YJ and YH implemented)
}

# We deal with the spectral channel individually.
list_index_ifu = [0, 10, 20]  # List of index to be used (nwl = 39 for SPHERE)
# You can also use all of them (np.arange(39)).

# You can first check which channel you will use
wave = amical.get_infos_obs.get_ifu_table(
    list_index_ifu, filtname=params_ami["filtname"]
)
print("Wave used:", wave, "µm")

datadir = "NRM_DATA/"

file_ifs = os.path.join(datadir, "example_sphere_ifs.fits")

# Then, we can clean and extract each wavelengths
l_cal = []
for i_wl in list_index_ifu:
    # Step 1: clean the data (only one wave at a time: i_wl index)
    cube_clean = amical.select_clean_data(file_ifs, i_wl=i_wl, **clean_param)

    # Step 2: extract data (i_wl must be specified to use the good bandwidth and
    # central wavelength (automatical in AMICAL).
    bs_ifs = amical.extract_bs(
        cube_clean,
        file_ifs,
        i_wl=i_wl,
        display=True,
        **params_ami,
    )

    # We convert the raw variable into appropriate format for visualisation
    cal = amical.oifits.wrap_raw(bs_ifs)

    # Or apply the standard calibration procedure using a calibrator
    # cal - amical.calibrate(bs_t, bs_c)  # See doc/example_SPHERE.py

    # We save all wavelenght into the same list
    l_cal.append(cal)

# Finally, you can display your observable
amical.show(l_cal)

# You can finaly save the file as usual
# amical.save(l_cal, oifits_file="fake_ifs.oifits", pa=bs_t.infos.pa)
