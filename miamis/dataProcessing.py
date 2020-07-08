import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from termcolor import cprint
from tqdm import tqdm

from miamis.tools import applyMaskApod, checkRadiusResize, crop_max


def ApplyPatchGhost(cube, xc, yc, radius=20, dx=0, dy=-200, method='bg'):
    """Apply a patch on an eventual artifacts/ghosts on the spectral filter (i.e.
    K1 filter of SPHERE presents an artifact/ghost at (392, 360)).

    Arguments:
    ----------
    `cube` {array} -- Data cube,\n
    `xc` {int} -- x-axis position of the artifact,\n
    `yc` {int} -- y-axis position of the artifact.

    Keyword Arguments:
    ----------
    `radius` {int} -- Radius to apply the patch in a circle (default: {10}),\n
    `dy` {int} -- Offset pixel number to compute background values (default: {0}),\n
    `dx` {int} -- Same along y-axis (default: {0}),\n
    `method` {str} -- If 'bg', the replacement values are the background computed at
    xc+dx, yx+dy, else zero is apply (default: {'bg'}).
    """
    cube_corrected = []
    for i in range(len(cube)):
        imA = cube[i].copy()
        isz = imA.shape[0]
        xc_off, yc_off = xc+dx, yc+dy
        xx, yy = np.arange(isz), np.arange(isz)
        xx_c = (xx-xc)
        yy_c = (yc-yy)
        xx_off = (xx-xc_off)
        yy_off = (yc_off-yy)
        distance = np.sqrt(xx_c**2 + yy_c[:, np.newaxis]**2)
        distance_off = np.sqrt(xx_off**2 + yy_off[:, np.newaxis]**2)
        cond_patch = (distance <= radius)
        cond_bg = (distance_off <= radius)
        if method == 'bg':
            imA[cond_patch] = imA[cond_bg]
        elif method == 'zero':
            imA[cond_patch] = 0
        cube_corrected.append(imA)
    cube_corrected = np.array(cube_corrected)
    return cube_corrected


def checkDataCube(cube, clip_fact=0.5, clip=False, verbose=True, display=True):
    """ Check the cleaned data cube using the position of the maximum in the
    fft image (supposed to be zero). If not in zero position, the fram is 
    rejected.

    Parameters:
    -----------
    `cube` {array} -- Data cube,\n
    `clip_fact` {float} -- Relative sigma if rejecting frames by 
    sigma-clipping (default=False),\n
    `clip` {bool} -- If True, sigma-clipping is used,\n
    `verbose` {bool} -- If True, print informations in the terminal,\n
    `display` {bool} -- If True, plot figures.


    """
    fft_fram = abs(np.fft.fft2(cube))
    # flag_fram, cube_flagged, cube_cleaned_checked = [], [], []

    fluxes, flag_fram, good_fram = [], [], []
    for i in range(len(fft_fram)):
        fluxes.append(fft_fram[i][0, 0])
        pos_max = np.argmax(fft_fram[i])
        if pos_max != 0:
            flag_fram.append(i)
        else:
            good_fram.append(cube[i])

    fluxes = np.array(fluxes)
    flag_fram = np.array(flag_fram)

    best_fr = np.argmax(fluxes)
    worst_fr = np.argmin(fluxes)

    std_flux = np.std(fluxes)
    med_flux = np.median(fluxes)

    if verbose:
        if (med_flux/std_flux) <= 5.:
            cprint('\nStd of the fluxes along the cube < 5 (%2.1f):\n -> sigma clipping is suggested (clip=True).' % (
                (med_flux/std_flux)), 'cyan')

    limit_flux = med_flux - clip_fact*std_flux

    if clip:
        cond_clip = (fluxes > limit_flux)
        cube_cleaned_checked = cube[cond_clip]
        ind_clip = np.where(fluxes <= limit_flux)[0]
    else:
        cube_cleaned_checked = np.array(good_fram)

    ind_clip2 = np.where(fluxes <= limit_flux)[0]
    if ((worst_fr in ind_clip2) and clip) or (worst_fr in flag_fram):
        ext = '(rejected)'
    else:
        ext = ''

    diffmm = 100*abs(np.max(fluxes) - np.min(fluxes))/med_flux
    if display:
        plt.figure()
        plt.plot(fluxes, label=r'|$\Delta F$|/$\sigma_F$=%2.0f (%2.2f %%)' %
                 (med_flux/std_flux, diffmm))
        if len(flag_fram) > 0:
            plt.scatter(flag_fram, fluxes[flag_fram],
                        s=52, facecolors='none', edgecolors='r', label='Rejected frames (maximum fluxes)')
        if clip:
            if len(ind_clip) > 0:
                plt.plot(ind_clip, fluxes[ind_clip], 'rx',
                         label='Rejected frames (clipping)')
            else:
                print('0')
        plt.hlines(limit_flux, 0, len(fluxes), lw=1,
                   ls='--', label='Clipping limit', zorder=10)
        plt.legend(loc='best', fontsize=9)
        plt.ylabel('Flux [counts]')
        plt.xlabel('# frames')
        plt.grid(alpha=.2)
        plt.tight_layout()

        plt.figure(figsize=(7, 7))
        plt.subplot(2, 2, 1)
        plt.title('Best fram (%i)' % best_fr)
        plt.imshow(cube[best_fr], norm=PowerNorm(.5), cmap='afmhot', vmin=0)
        plt.subplot(2, 2, 2)
        plt.imshow(np.fft.fftshift(fft_fram[best_fr]), cmap='gist_stern')
        plt.subplot(2, 2, 3)
        plt.title('Worst fram (%i) %s' % (worst_fr, ext))
        plt.imshow(cube[worst_fr], norm=PowerNorm(.5), cmap='afmhot', vmin=0)
        plt.subplot(2, 2, 4)
        plt.imshow(np.fft.fftshift(fft_fram[worst_fr]), cmap='gist_stern')
        plt.tight_layout()
    if verbose:
        n_good = len(cube_cleaned_checked)
        n_bad = len(cube) - n_good
        if clip:
            cprint('\n---- σ-clip + centered fluxes selection ---', 'cyan')
        else:
            cprint('\n---- centered fluxes selection ---', 'cyan')
        print('%i/%i (%2.1f%%) are flagged as bad frames' %
              (n_bad, len(cube), 100*float(n_bad)/len(cube)))
    return cube_cleaned_checked


def skyCorrection(imA, r1=100, dr=20, verbose=False):
    """
    Perform background sky correction to be as close to zero as possible.
    """
    isz = imA.shape[0]
    xc, yc = isz//2, isz//2
    xx, yy = np.arange(isz), np.arange(isz)
    xx2 = (xx-xc)
    yy2 = (yc-yy)
    r2 = r1 + dr

    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)
    cond_bg = (r1 <= distance) & (distance <= r2)

    try:
        minA = imA.min()
        imB = imA + 1.01*abs(minA)
        backgroundB = np.mean(imB[cond_bg])
        imC = imB - backgroundB
        backgroundC = np.mean(imC[cond_bg])
    except IndexError:
        imC = imA.copy()
        backgroundC = 0
        if verbose:
            cprint('Warning: Background not computed', 'green')
            cprint(
                '-> check the inner and outer radius rings (checkrad option).', 'green')

    return imC, backgroundC


def clean_data(data, isz=None, r1=None, dr=None, edge=100, n_show=0, checkrad=False, verbose=False):
    """ Clean data (if not simulated data).

    Parameters:
    -----------

    `data` {np.array} -- datacube containing the NRM data\n
    `isz` {int} -- Size of the cropped image (default: {None})\n
    `r1` {int} -- Radius of the rings to compute background sky (default: {None})\n
    `dr` {int} -- Outer radius to compute sky (default: {None})\n
    `edge` {int} -- Patch the edges of the image (VLT/SPHERE artifact, default: {200}),\n
    `checkrad` {bool} -- If True, check the resizing and sky substraction parameters (default: {False})\n

    Returns:
    --------
    `cube` {np.array} -- Cleaned datacube.
    """
    if data.shape[1] % 2 == 1:
        data = np.array([im[:-1, :-1] for im in data])

    n_im = data.shape[0]
    if checkrad:
        img0 = data[n_show]
        if edge != 0:
            img0[:, 0:edge] = 0
            img0[:, -edge:-1] = 0
            img0[0:edge, :] = 0
            img0[-edge:-1, :] = 0
        ref0_max, pos = crop_max(img0, isz, f=3)
        fig = checkRadiusResize(img0, isz, r1, dr, pos)
        fig.show()
        return None

    cube = []
    for i in tqdm(range(n_im), ncols=100, desc='Cleaning', leave=False):
        # img0 = applyMaskApod(data[i], r=int(npix//3))
        img0 = data[i]
        if edge != 0:
            img0[:, 0:edge] = 0
            img0[:, -edge:-1] = 0
            img0[0:edge, :] = 0
            img0[-edge:-1, :] = 0
        im_rec_max, pos = crop_max(img0, isz, f=3)
        img_biased, bg = skyCorrection(im_rec_max, r1=r1, dr=dr)

        try:
            img = applyMaskApod(img_biased, r=isz//3)
            cube.append(img)
        except ValueError:
            if verbose:
                cprint(
                    'Error: problem with centering process -> check isz/r1/dr parameters.', 'red')
                cprint(i, 'red')

    cube = np.array(cube)
    return cube


def selectCleanData(filename, isz=256, r1=100, dr=10, edge=100, clip=True,
                    clip_fact=0.5, checkrad=False, n_show=0, corr_ghost=True,
                    verbose=False, display=False):
    """ Clean and select good datacube (sigma-clipping using fluxes variations)."""
    hdu = fits.open(filename)
    cube = hdu[0].data
    hdr = hdu[0].header

    if hdr['INSTRUME'] == 'SPHERE':
        seeing_start = float(hdr['HIERARCH ESO TEL AMBI FWHM START'])
        seeing = float(hdr['HIERARCH ESO TEL IA FWHM'])
        seeing_end = float(hdr['HIERARCH ESO TEL AMBI FWHM END'])

    if verbose:
        print('\n----- Seeing conditions -----')
        print("%2.2f (start), %2.2f (end), %2.2f (Corrected AirMass)" %
              (seeing_start, seeing_end, seeing))

    if corr_ghost:
        if (hdr['INSTRUME'] == 'SPHERE') & (hdr['FILTER'] == 'K1'):
            cube_patched = ApplyPatchGhost(cube, 392, 360)
        elif (hdr['INSTRUME'] == 'SPHERE') & (hdr['FILTER'] == 'K2'):
            cube_patched = ApplyPatchGhost(cube, 378, 311)
            cube_patched = ApplyPatchGhost(cube_patched, 891, 315)
    else:
        cube_patched = cube.copy()

    cube_cleaned = clean_data(cube_patched, isz=isz, r1=r1, edge=edge,
                              dr=dr, n_show=n_show, checkrad=checkrad,
                              verbose=verbose)

    if cube_cleaned is None:
        return None

    cube_final = checkDataCube(cube_cleaned, clip=clip, clip_fact=clip_fact,
                               verbose=verbose, display=display)
    return cube_final, cube_cleaned
