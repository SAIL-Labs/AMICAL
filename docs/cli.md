# Command-Line Interface

AMICAL provides a command-line interface (CLI) to reduce data directly from the terminal.

```bash
$ amical -h

usage: amical [-h] {clean,extract,calibrate} ...

positional arguments:
  {clean,extract,calibrate}
    clean               Clean reduced data for NRM extraction.
    extract             Extract the bispectrum from cleaned NRM data.
    calibrate           Calibrate the extracted NRM data with the associated calibrator.

options:
  -h, --help            show this help message and exit
```

Each of the three subcommands (`clean`, `extract`, `calibrate`)
performs one of the main steps discussed in [Getting Started](../getting-started/).

Below we show the help message for each subcommand.

## Cleaning data

```bash
$ amical clean -h

usage: amical clean [-h] [--datadir DATADIR] [--outdir OUTDIR] [--isz ISZ] [--r1 R1] [--dr DR] [--apod] [--window WINDOW] [--sky] [--clip] [--kernel KERNEL] [-c] [-p] [-f FILE] [-a]

options:
  -h, --help         show this help message and exit
  --datadir DATADIR  Repository containing the reduced NRM data (default: data/).
  --outdir OUTDIR    Repository to save the cleaned NRM data (as fits files)(default: cleaned/).
  --isz ISZ          Size of the cropped image [pix] (default: None).
  --r1 R1            Radius of the rings to compute background sky [pix] (default: 32).
  --dr DR            Outer radius to compute sky (r2=r1+dr) [pix] (default: 3).
  --apod             Perform apodisation using a super-gaussian function (known as windowing). The gaussian FWHM is set by the parameter `window`.
  --window WINDOW    FWHM used for windowing (used with --apod)(default: 65)
  --sky              Remove sky background using the annulus technique(computed between r1 and r1 + dr)
  --clip             Perform sigma-clipping to reject bad frames (default: False).
  --kernel KERNEL    kernel size used in the applied median filter (to find the center)(default: 3)
  -c, --check        Check the cleaning parameters (plot relevant radius in the image).
  -p, --plot         Plot the diagnostic figures.
  -f, --file FILE    Select the file index to be clean (default allows user selection).
  -a, --all          Clean all data files in --datadir.
~/repos/astro/AMICAL   update-docs ?1                                                                                                                                                                                             amical
```

## Extracting observables

```bash
$ amical extract -h

usage: amical extract [-h] [--datadir DATADIR] [--outdir OUTDIR] [--maskname MASKNAME] [--peakmethod {fft,gauss,unique,square}] [--instrum INSTRUM] [--targetname TARGETNAME] [--filtname FILTNAME] [--nwl NWL] [--cutoff CUTOFF]
                      [--diam DIAM] [--fw FW] [--multitri] [--unbias] [--thetadet THETADET] [--scaling SCALING] [--iwl IWL] [--save_to SAVE_TO] [-p] [-e] [-f FILE] [-a]

options:
  -h, --help            show this help message and exit
  --datadir DATADIR     Repository containing the cleaned NRM data (default: cleaned/).
  --outdir OUTDIR       Repository to save the extracted bispectrum (as hdf5 .h5 files)(default: extracted/).
  --maskname MASKNAME   Name of the mask aperture (default: g7).
  --peakmethod {fft,gauss,unique,square}
                        Fourier sampling method (default: fft).
  --instrum INSTRUM     Name of the instrument (if not found in the header).
  --targetname TARGETNAME
                        Name of the target (if not found in the header).
  --filtname FILTNAME   Name of the spectral filter (if not found in the header).
  --nwl NWL             Number of elements to sample the spectral filters (default: 3).
  --cutoff CUTOFF       Cutoff limit between noise and signal for fft method (default: 0.0001).
  --diam DIAM           Diameter of a single aperture (default: 0.8).
  --fw FW               Relative size of the splodge used to compute multiple triangle indices and the fwhm of the 'gauss' technique (default: 0.7).
  --multitri            Compute the CP over multiple triangles (Monnier method).
  --unbias              Unbias the V2 using the Fourier base.
  --thetadet THETADET   Angle [deg] to rotate the mask compare to the detector (if the mask is notperfectly aligned with the detector, e.g.: VLT/VISIR) (default: 0).
  --scaling SCALING     Scaling factor to be applied to match the mask with data (e.g.: VAMPIRES) (default: 1).
  --iwl IWL             Only used for IFU data (e.g.: IFS/SPHERE), select the desired spectral channel to retrieve the appropriate wavelength and mask positions.
  --save_to SAVE_TO     If save_to is set, figures are saved within it as pdf (default: None).
  -p, --plot            Plot the diagnostic figures.
  -e, --expert          Save additional plots.
  -f, --file FILE       Select the file index to be extracted (default allows user selection).
  -a, --all             Extract bispectrum from all data files in --datadir.
```

## Calibrating observables

```bash
$ amical calibrate -h

usage: amical calibrate [-h] [--datadir DATADIR] [--outdir OUTDIR] [--clip] [--norm] [--phscorr] [--atmcorr] [-p]

options:
  -h, --help         show this help message and exit
  --datadir DATADIR  Repository containing the extracted bispectrum (default: extracted/).
  --outdir OUTDIR    Repository to save the calibrated oifits files (default: calibrated/).
  --clip             Sigma clipping is performed over the calibrator files (if any) to reject bad observables due to seeing conditions, centering, etc (default: False).
  --norm             CP uncertaintities are normalized by np.sqrt(n_holes/3.) to not over use the non-independant closure phases (default: False).
  --phscorr          Apply a phasor correction due to piston between holes (default: False).
  --atmcorr          Apply a atmospheric correction on V2 from seeing and wind shacking issues (default: False).
  -p, --plot         Plot the calibrated data.
```
