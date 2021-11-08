from argparse import ArgumentParser
from typing import List
from typing import Optional

from amical._cli.commands.calibrate import perform_calibrate
from amical._cli.commands.clean import perform_clean
from amical._cli.commands.extract import perform_extract


def main(argv: Optional[List[str]] = None) -> int:
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    # __________________________________________________________________________
    # CLEANING STEP
    # __________________________________________________________________________

    clean_parser = subparsers.add_parser(
        "clean", help="Clean reduced data for NRM extraction."
    )

    # INPUT and OUTPUT directory
    # __________________________________________________________________________

    clean_parser.add_argument(
        "--datadir",
        default="data/",
        help="Repository containing the reduced NRM data (default: %(default)s).",
    )
    clean_parser.add_argument(
        "--outdir",
        default="cleaned/",
        help="Repository to save the cleaned NRM data (as fits files)(default: %(default)s).",
    )

    # Cleaning parameters of AMICAL
    # __________________________________________________________________________

    clean_parser.add_argument(
        "--isz",
        default=None,
        type=int,
        help="Size of the cropped image [pix] (default: %(default)s).",
    )
    clean_parser.add_argument(
        "--r1",
        default=32,
        type=int,
        help="Radius of the rings to compute background sky [pix] (default: %(default)s).",
    )
    clean_parser.add_argument(
        "--dr",
        default=3,
        type=int,
        help="Outer radius to compute sky (r2=r1+dr) [pix] (default: %(default)s).",
    )
    clean_parser.add_argument(
        "--apod",
        action="store_true",
        help="Perform apodisation using a super-gaussian function "
        + "(known as windowing)."
        + " The gaussian FWHM is set by the parameter `window`.",
    )
    clean_parser.add_argument(
        "--window",
        default=65,
        type=int,
        help="FWHM used for windowing (used with --apod)(default: %(default)s)",
    )
    clean_parser.add_argument(
        "--sky",
        action="store_true",
        help="Remove sky background using the annulus technique"
        + "(computed between r1 and r1 + dr)",
    )
    clean_parser.add_argument(
        "--clip",
        action="store_true",
        help="Perform sigma-clipping to reject bad frames (default: %(default)s).",
    )
    clean_parser.add_argument(
        "--kernel",
        default=3,
        type=int,
        help="kernel size used in the applied median filter (to find the center)(default: %(default)s)",
    )

    # CLI parameters
    # __________________________________________________________________________

    clean_parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Check the cleaning parameters (plot relevant radius in the image).",
    )
    clean_parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the diagnostic figures.",
    )
    clean_parser.add_argument(
        "-f",
        "--file",
        default=-1,
        type=int,
        help="Select the file index to be clean (default allows user selection).",
    )
    clean_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Clean all data files in --datadir.",
    )

    # __________________________________________________________________________
    # EXTRACT STEP
    # __________________________________________________________________________

    extract_parser = subparsers.add_parser(
        "extract", help="Extract the bispectrum from cleaned NRM data."
    )
    # INPUT and OUTPUT directory
    # __________________________________________________________________________

    extract_parser.add_argument(
        "--datadir",
        default="cleaned/",
        help="Repository containing the cleaned NRM data (default: %(default)s).",
    )
    extract_parser.add_argument(
        "--outdir",
        default="extracted/",
        help="Repository to save the extracted bispectrum (as hdf5 .h5 files)(default: %(default)s).",
    )

    # Extracting parameters of AMICAL
    # __________________________________________________________________________

    #  ## IMPORTANT PARAMS ##
    extract_parser.add_argument(
        "--maskname",
        default="g7",
        help="Name of the mask aperture (default: %(default)s).",
    )
    extract_parser.add_argument(
        "--peakmethod",
        choices=["fft", "gauss", "unique", "square"],
        default="fft",
        help="Fourier sampling method (default: %(default)s).",
    )

    # User params if missing informations from the header
    extract_parser.add_argument(
        "--instrum",
        default=None,
        help="Name of the instrument (if not found in the header).",
    )
    extract_parser.add_argument(
        "--targetname",
        default=None,
        help="Name of the target (if not found in the header).",
    )
    extract_parser.add_argument(
        "--filtname",
        default=None,
        help="Name of the spectral filter (if not found in the header).",
    )

    # ## FOLLOWING PARAMETERS ARE RARELY CHANGED
    extract_parser.add_argument(
        "--nwl",
        default=3,
        type=int,
        help="Number of elements to sample the spectral filters (default: %(default)s).",
    )
    extract_parser.add_argument(
        "--cutoff",
        default=1e-4,
        help="Cutoff limit between noise and signal for fft method (default: %(default)s).",
    )
    extract_parser.add_argument(
        "--diam",
        default=0.8,
        help="Diameter of a single aperture (default: %(default)s).",
    )
    extract_parser.add_argument(
        "--fw",
        default=0.7,
        type=float,
        help="Relative size of the splodge used to compute multiple triangle indices "
        + " and the fwhm of the 'gauss' technique (default: %(default)s).",
    )
    extract_parser.add_argument(
        "--multitri",
        action="store_true",
        help="Compute the CP over multiple triangles (Monnier method).",
    )
    extract_parser.add_argument(
        "--unbias",
        action="store_false",
        help="Unbias the V2 using the Fourier base.",
    )

    # Parameters to rotate and centrally-enlarge the mask position.
    extract_parser.add_argument(
        "--thetadet",
        default=0,
        type=float,
        help="Angle [deg] to rotate the mask compare to the detector (if the mask is not"
        + "perfectly aligned with the detector, e.g.: VLT/VISIR) (default: %(default)s).",
    )
    extract_parser.add_argument(
        "--scaling",
        default=1,
        type=float,
        help="Scaling factor to be applied to match the mask with data (e.g.: VAMPIRES) (default: %(default)s).",
    )
    extract_parser.add_argument(
        "--iwl",
        default=None,
        type=int,
        help="Only used for IFU data (e.g.: IFS/SPHERE), select the desired spectral"
        + " channel to retrieve the appropriate wavelength and mask positions.",
    )

    # CLI parameters
    # __________________________________________________________________________

    extract_parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        help="If save_to is set, figures are saved within it as pdf (default: %(default)s).",
    )
    extract_parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the diagnostic figures.",
    )
    extract_parser.add_argument(
        "-e",
        "--expert",
        action="store_true",
        help="Save additional plots.",
    )
    extract_parser.add_argument(
        "-f",
        "--file",
        default=-1,
        type=int,
        help="Select the file index to be extracted (default allows user selection).",
    )
    extract_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Extract bispectrum from all data files in --datadir.",
    )

    # __________________________________________________________________________
    # CALIBRATE STEP
    # __________________________________________________________________________

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate the extracted NRM data with the associated calibrator.",
    )
    # INPUT and OUTPUT directory
    # __________________________________________________________________________

    calibrate_parser.add_argument(
        "--datadir",
        default="extracted/",
        help="Repository containing the extracted bispectrum (default: %(default)s).",
    )
    calibrate_parser.add_argument(
        "--outdir",
        default="calibrated/",
        help="Repository to save the calibrated oifits files (default: %(default)s).",
    )

    #  ## IMPORTANT PARAMS ##
    calibrate_parser.add_argument(
        "--clip",
        action="store_true",
        help="Sigma clipping is performed over the calibrator files (if any) to reject "
        + "bad observables due to seeing conditions, centering, etc (default: %(default)s).",
    )
    calibrate_parser.add_argument(
        "--norm",
        action="store_true",
        help="CP uncertaintities are normalized by np.sqrt(n_holes/3.) to not "
        + "over use the non-independant closure phases (default: %(default)s).",
    )
    calibrate_parser.add_argument(
        "--phscorr",
        action="store_true",
        help="Apply a phasor correction due to piston between holes (default: %(default)s).",
    )
    calibrate_parser.add_argument(
        "--atmcorr",
        action="store_true",
        help="Apply a atmospheric correction on V2 from seeing and wind shacking issues (default: %(default)s).",
    )

    # CLI parameters
    # __________________________________________________________________________
    calibrate_parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the calibrated data.",
    )

    args = parser.parse_args(argv)

    retv = 1
    if args.command == "clean":
        retv = perform_clean(args)
    elif args.command == "extract":
        retv = perform_extract(args)
    elif args.command == "calibrate":
        retv = perform_calibrate(args)
    return retv


if __name__ == "__main__":
    exit(main())
