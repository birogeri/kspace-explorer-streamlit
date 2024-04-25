import streamlit as st
import PIL
import numpy as np
import pydicom
import io
from PIL import Image

st.set_page_config(
    page_title='K-space Explorer',
    page_icon='images/icon.ico', layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': "https://github.com/birogeri/kspace-explorer-streamlit/issues",
        'About': "# K-space Explorer\n"
                 "### Online Demo\n"
                 "K-space Explorer is a free and "
                 "open-source educational tool primarily for students and "
                 "MRI radiographers.\n"
                 "This online tool has a sample of the "
                 "features offered by the desktop application.\n\n"
                 "Homepage: [k-space.app](https://k-space.app)\n\n"
                 "Author & Contributors: [View on GitHub](https://github.com/birogeri/kspace-explorer#author--contributors)\n "
                 "> In memoriam Miklós Derváli"
    })

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift


@st.cache_data
def open_file(file, dtype: np.dtype = np.float32) -> np.ndarray:
    """Tries to load image data into a NumPy ndarray

    The function first tries to use the PIL Image library to identify and load
    the image. PIL will convert the image to 8-bit pixels, black and white.
    If PIL fails pydicom is the next choice.

    Parameters:
        file (str): Path or StreamLit UploadedFile object
        dtype (np.dtype): image array dtype (e.g. np.float64)

    Returns:
        np.ndarray: a floating point NumPy ndarray of the specified dtype
    """

    try:
        with Image.open(file) as f:
            img_file = f.convert('F')  # 'F' mode: 32-bit floating point pixels
            img_pixel_array = np.array(img_file).astype(dtype)
        return img_pixel_array
    except FileNotFoundError as e:
        if 'im' not in globals():   # Quit gracefully if first start fails
            st.exception(e)
    except PIL.UnidentifiedImageError:
        try:
            if isinstance(file, io.BytesIO):
                # when the file object is an uploaded file (presumably DICOM)
                file.seek(0)
                with pydicom.dcmread(file) as dcm_file:
                    img_pixel_array = dcm_file.pixel_array.astype(dtype)
            else:
                # when the file object is a string (most commonly the default file)
                with pydicom.dcmread(file) as dcm_file:
                    img_pixel_array = dcm_file.pixel_array.astype(dtype)
            img_pixel_array.setflags(write=True)
            return img_pixel_array
        except Exception as e:
            st.exception(e)


class ImageManipulators:
    """A class that contains a 2D image and kspace pair and modifier methods

    This class will load the specified image or raw data and performs any
    actions that modify the image or kspace data. A new instance should be
    initialized for new images.
    """

    def __init__(self, pixel_data: np.ndarray):
        """Opening the image and initializing variables based on image size

        Parameters:
            pixel_data (np.ndarray): 2D pixel data of image or kspace
        """

        self.img = pixel_data.copy()
        self.kspacedata = np.zeros_like(self.img, dtype=np.complex64)
        self.image_display_data = np.require(self.img, np.uint8, 'C')
        self.kspace_display_data = np.zeros_like(self.image_display_data)
        self.orig_kspacedata = np.zeros_like(self.kspacedata)
        self.kspace_abs = np.zeros_like(self.kspacedata, dtype=np.float32)
        self.noise_map = np.zeros_like(self.kspace_abs)
        self.signal_to_noise = 30
        self.spikes = []
        self.patches = []

        self.np_fft(self.img, self.kspacedata)

        self.orig_kspacedata[:] = self.kspacedata  # Store data write-protected
        self.orig_kspacedata.setflags(write=False)

        self.prepare_displays()

    @staticmethod
    def np_ifft(kspace: np.ndarray, out: np.ndarray):
        """Performs inverse FFT function (kspace to [magnitude] image)

        Performs iFFT on the input data and updates the display variables for
        the image domain (magnitude) image and the kspace as well.

        Parameters:
            kspace (np.ndarray): Complex kspace ndarray
            out (np.ndarray): Array to store values
        """
        np.absolute(fftshift(ifft2(ifftshift(kspace))), out=out)

    @staticmethod
    def np_fft(img: np.ndarray, out: np.ndarray):
        """ Performs FFT function (image to kspace)

        Performs FFT function, FFT shift and stores the unmodified kspace data
        in a variable and also saves one copy for display and edit purposes.

        Parameters:
            img (np.ndarray): The NumPy ndarray to be transformed
            out (np.ndarray): Array to store output (must be same shape as img)
        """
        out[:] = fftshift(fft2(ifftshift(img)))

    @staticmethod
    def normalise(f: np.ndarray):
        """ Normalises array by "streching" all values to be between 0-255.

        Parameters:
            f (np.ndarray): input array
        """
        fmin = float(np.min(f))
        fmax = float(np.max(f))
        if fmax != fmin:
            coeff = fmax - fmin
            f[:] = np.floor((f[:] - fmin) / coeff * 255.)

    @staticmethod
    def apply_window(f: np.ndarray, window_val: dict = None):
        """ Applies window values to the array

        Excludes certain values based on window width and center before
        applying normalisation on array f.
        Window values are interpreted as percentages of the maximum
        intensity of the actual image.
        For example if window_val is 1, 0.5 and image has maximum intensity
        of 196 then window width is 196, window center is 98.
        Code applied from contrib-pydicom see license below:
            Copyright (c) 2009 Darcy Mason, Adit Panchal
            This file is part of pydicom, relased under an MIT license.
            See the file LICENSE included with this distribution, also
            available at https://github.com/pydicom/pydicom
            Based on image.py from pydicom version 0.9.3,
            LUT code added by Adit Panchal

        Parameters:
            f (np.ndarray): the array to be windowed
            window_val (dict): window width and window center dict
        """
        fmax = np.max(f)
        fmin = np.min(f)
        if fmax != fmin:
            ww = (window_val['ww'] * fmax) if window_val else fmax
            wc = (window_val['wc'] * fmax) if window_val else (ww / 2)
            w_low = wc - ww / 2
            w_high = wc + ww / 2
            f[:] = np.piecewise(f, [f <= w_low, f > w_high], [0, 255,
                                lambda x: ((x - wc) / ww + 0.5) * 255])

    def prepare_displays(self, kscale: int = -3, lut: dict = None):
        """ Prepares kspace and image for display in the user interface

        Magnitude of the kspace is taken and scaling is applied for display
        purposes. This scaled representation is then transformed to a 256 color
        grayscale image by normalisation (where the highest and lowest
        intensity pixels will be intensity level 255 and 0 respectively)
        Similarly the image is prepared with the addition of windowing
        (excluding certain values based on user preference before normalisation
        e.g. intensity lower than 20 and higher than 200).

        Parameters:
            kscale (int): kspace intensity scaling constant (10^kscale)
            lut (dict): window width and window center dict
        """

        # 1. Apply window to image
        self.apply_window(self.img, lut)

        # 2. Prepare kspace display - get magnitude then scale and normalise
        # K-space scaling: https://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
        np.absolute(self.kspacedata, out=self.kspace_abs)
        if np.any(self.kspace_abs):
            scaling_c = np.power(10., kscale)
            np.log1p(self.kspace_abs * scaling_c, out=self.kspace_abs)
            self.normalise(self.kspace_abs)

        # 3. Obtain uint8 type arrays for QML display
        self.image_display_data[:] = np.require(self.img, np.uint8)
        self.kspace_display_data[:] = np.require(self.kspace_abs, np.uint8)

    def resize_arrays(self, size: (int, int)):
        """ Resize arrays for image size changes (e.g. remove kspace lines etc.)

        Called by undersampling kspace and the image_change method. If the FOV
        is modified, image_change will reset the size based on the original
        kspace, performs other modifications to the image that are applied
        before undersampling and then reapplies the size change.

        Parameters:
            size (int, int): size of the new array
        """
        self.img.resize(size)
        self.image_display_data.resize(size)
        self.kspace_display_data.resize(size)
        self.kspace_abs.resize(size)
        self.kspacedata.resize(size, refcheck=False)

    @staticmethod
    def reduced_scan_percentage(kspace: np.ndarray, percentage: float):
        """Deletes a percentage of lines from the kspace in phase direction

        Deletes an equal number of lines from the top and bottom of kspace
        to only keep the specified percentage of sampled lines. For example if
        the image has 256 lines and percentage is 50.0 then 64 lines will be
        deleted from the top and bottom and 128 will be kept in the middle.

        Parameters:
            kspace (np.ndarray): Complex kspace data
            percentage (float): The percentage of lines sampled (0.0 - 100.0)
        """

        if int(percentage) < 100:
            percentage_delete = 1 - percentage / 100
            lines_to_delete = round(percentage_delete * kspace.shape[0] / 2)
            if lines_to_delete:
                kspace[0:lines_to_delete] = 0
                kspace[-lines_to_delete:] = 0

    @staticmethod
    def high_pass_filter(kspace: np.ndarray, radius: float):
        """High pass filter removes the low spatial frequencies from k-space

        This function deletes the center of kspace by removing values
        inside a circle of given size. The circle's radius is determined by
        the 'radius' float variable (0.0 - 100) as ratio of the lenght of
        the image diagonally.

        Parameters:
            kspace (np.ndarray): Complex kspace data
            radius (float): Relative size of the kspace mask circle (percent)
        """

        if radius > 0:
            r = np.hypot(*kspace.shape) / 2 * radius / 100
            rows, cols = np.array(kspace.shape, dtype=int)
            a, b = np.floor(np.array((rows, cols)) / 2).astype(int)
            y, x = np.ogrid[-a:rows - a, -b:cols - b]
            mask = x * x + y * y <= r * r
            kspace[mask] = 0

    @staticmethod
    def low_pass_filter(kspace: np.ndarray, radius: float):
        """Low pass filter removes the high spatial frequencies from k-space

        This function only keeps the center of kspace by removing values
        outside a circle of given size. The circle's radius is determined by
        the 'radius' float variable (0.0 - 100) as ratio of the lenght of
        the image diagonally

        Parameters:
            kspace (np.ndarray): Complex kspace data
            radius (float): Relative size of the kspace mask circle (percent)
        """

        if radius < 100:
            r = np.hypot(*kspace.shape) / 2 * radius / 100
            rows, cols = np.array(kspace.shape, dtype=int)
            a, b = np.floor(np.array((rows, cols)) / 2).astype(int)
            y, x = np.ogrid[-a:rows - a, -b:cols - b]
            mask = x * x + y * y <= r * r
            kspace[~mask] = 0

    @staticmethod
    def add_noise(kspace: np.ndarray, signal_to_noise: float,
                  current_noise: np.ndarray, generate_new_noise=False):
        """Adds random Guassian white noise to k-space

        Adds noise to the image to simulate an image with the given
        signal-to-noise ratio, so that SNR [dB] = 20log10(S/N)
        where S is the mean signal and N is the standard deviation of the noise.

        Parameters:
            kspace (np.ndarray): Complex kspace ndarray
            signal_to_noise (float): SNR in decibels (-30dB - +30dB)
            current_noise (np.ndarray): the existing noise map
            generate_new_noise (bool): flag to generate new noise map
        """

        if signal_to_noise < 30:
            if generate_new_noise:
                mean_signal = np.mean(np.abs(kspace))
                std_noise = mean_signal / np.power(10, (signal_to_noise / 20))
                current_noise[:] = std_noise * np.random.randn(*kspace.shape)
            kspace += current_noise

    @staticmethod
    def partial_fourier(kspace: np.ndarray, percentage: float, zf: bool):
        """ Partial Fourier

        Also known as half scan - only acquire a little over half of k-space
        or more and use conjugate symmetry to fill the rest.

        Parameters:
            kspace (np.ndarray): Complex k-space
            percentage (float): Sampled k-space percentage
            zf (bool): Zero-fill k-space instead of using symmetry
        """

        if int(percentage) != 100:
            percentage = 1 - percentage / 100
            rows_to_skip = round(percentage * (kspace.shape[0] / 2 - 1))
            if rows_to_skip and zf:
                # Partial Fourier (lines not acquired are filled with zeros)
                kspace[-rows_to_skip:] = 0
            elif rows_to_skip:
                # If the kspace has an even resolution then the
                # mirrored part will be shifted (k-space center signal
                # (DC signal) is off center). This determines the peak
                # position and adjusts the mirrored quadrants accordingly
                # https://www.ncbi.nlm.nih.gov/pubmed/22987283

                # Following two lines are a connoisseur's (== obscure) way of
                # returning 1 if the number is even and 0 otherwise. Enjoy!
                shift_hor = not kspace.shape[1] & 0x1  # Bitwise AND
                shift_ver = 0 if kspace.shape[0] % 2 else 1  # Ternary operator
                s = (shift_ver, shift_hor)

                # 1. Obtain a view of the array backwards (rotated 180 degrees)
                # 2. If the peak is off center horizontally (e.g. number of
                #       columns or rows is even) roll lines to realign the
                #       highest amplitude parts
                # 3. Do the same vertically
                kspace[-rows_to_skip:] = \
                    np.roll(kspace[::-1, ::-1], s, axis=(0, 1))[-rows_to_skip:]

                # Conjugate replaced lines
                np.conj(kspace[-rows_to_skip:], kspace[-rows_to_skip:])

    @staticmethod
    def hamming(kspace: np.ndarray):
        """ Hamming filter

        Applies a 2D Hamming filter to reduce Gibbs ringing
        References:
            https://mriquestions.com/gibbs-artifact.html
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4058219/
            https://www.roberthovden.com/tutorial/2015/fftartifacts.html

        Parameters:
            kspace: Complex k-space numpy.ndarray
        """
        x, y = kspace.shape
        window = np.outer(np.hamming(x), np.hamming(y))
        kspace *= window

    @staticmethod
    def undersample(kspace: np.ndarray, factor: int, compress: bool):
        """ Skipping every nth kspace line

        Simulates acquiring every nth (where n is the acceleration factor) line
        of kspace, starting from the midline. Commonly used in SENSE algorithm.

        Parameters:
            kspace: Complex k-space numpy.ndarray
            factor: Only scan every nth line (n=factor) starting from midline
            compress: compress kspace by removing empty lines (rectangular FOV)
        """
        # TODO memory optimise this (kspace sized memory created 3 times)
        if factor > 1:
            mask = np.ones(kspace.shape, dtype=bool)
            midline = kspace.shape[0] // 2
            mask[midline::factor] = 0
            mask[midline::-factor] = 0
            if compress:
                q = kspace[~mask]
                q = q.reshape(q.size // kspace.shape[1], kspace.shape[1])
                im.resize_arrays(q.shape)
                kspace[:] = q[:]
            else:
                kspace[mask] = 0

    @staticmethod
    def decrease_dc(kspace: np.ndarray, percentage: int):
        """Decreases the highest peak in kspace (DC signal)

        Parameters:
            kspace: Complex k-space numpy.ndarray
            percentage: reduce the DC value by this value
        """
        x = kspace.shape[0] // 2
        y = kspace.shape[1] // 2
        kspace[x, y] *= (100 - percentage) / 100

    @staticmethod
    def apply_spikes(kspace: np.ndarray, spikes: list):
        """Overlays spikes to kspace

        Apply spikes (max value pixels) to the kspace data at the specified
        coordinates.

        Parameters:
            kspace (np.ndarray): Complex kspace ndarray
            spikes (list): coordinates for the spikes (row, column)
        """
        spike_intensity = np.max(kspace) * 2
        for spike in spikes:
            kspace[spike] = spike_intensity

    @staticmethod
    def apply_patches(kspace, patches: list):
        """Applies patches to kspace

         Apply patches (zero value squares) to the kspace data at the
         specified coordinates and size.

         Parameters:
             kspace (np.ndarray): Complex kspace ndarray
             patches (list): coordinates for the spikes (row, column, radius)
         """
        for patch in patches:
            x, y, size = patch[0], patch[1], patch[2]
            kspace[max(x - size, 0):x + size + 1,
                   max(y - size, 0):y + size + 1] = 0

    @staticmethod
    def filling(kspace: np.ndarray, value: float, mode: int):
        """Receives kspace filling UI changes and redirects to filling methods

        When the kspace filling simulation slider changes or simulation plays,
        this method receives the acquision phase (value: float, 0-100%)

        Parameters:
            kspace (np.ndarray): Complex kspace ndarray
            value (float): acquisition phase in percent
            mode (int): kspace filling mode
        """
        if mode == 0:  # Linear filling
            im.filling_linear(kspace, value)
        elif mode == 1:  # Centric filling
            im.filling_centric(kspace, value)
        elif mode == 2:  # Single shot EPI blipped
            im.filling_ss_epi_blipped(kspace, value)
        elif mode == 3:  # Archimedean spiral
            # im.filling_spiral(kspace, value)
            pass

    @staticmethod
    def filling_linear(kspace: np.ndarray, value: float):
        """Linear kspace filling

        Starts with the top left corner and sequentially fills kspace from
        top to bottom
        Parameters:
            kspace (np.ndarray): Complex kspace ndarray
            value (float): acquisition phase in percent
        """
        kspace.flat[int(kspace.size * value // 100)::] = 0

    @staticmethod
    def filling_centric(kspace: np.ndarray, value: float):
        """ Centric filling method

        Fills the center line first from left to right and then alternating one
        line above and one below.
        """
        ksp_centric = np.zeros_like(kspace)

        # reorder
        ksp_centric[0::2] = kspace[kspace.shape[0] // 2::]
        ksp_centric[1::2] = kspace[kspace.shape[0] // 2 - 1::-1]

        ksp_centric.flat[int(kspace.size * value / 100)::] = 0

        # original order
        kspace[(kspace.shape[0]) // 2 - 1::-1] = ksp_centric[1::2]
        kspace[(kspace.shape[0]) // 2::] = ksp_centric[0::2]

    @staticmethod
    def filling_ss_epi_blipped(kspace: np.ndarray, value: float):
        # Single-shot blipped EPI (zig-zag pattern)
        # https://www.imaios.com/en/e-Courses/e-MRI/MRI-Sequences/echo-planar-imaging
        ksp_epi = np.zeros_like(kspace)
        ksp_epi[::2] = kspace[::2]
        ksp_epi[1::2] = kspace[1::2, ::-1]  # Every second line backwards

        ksp_epi.flat[int(kspace.size * value / 100)::] = 0

        kspace[::2] = ksp_epi[::2]
        kspace[1::2] = ksp_epi[1::2, ::-1]


def image_change(s):
    """ Apply kspace modifiers to kspace and get resulting image"""

    # Get a copy of the original k-space data to play with
    im.resize_arrays(im.orig_kspacedata.shape)
    im.kspacedata[:] = im.orig_kspacedata

    # 01 - Noise
    if 'noise_value' in s:
        new_snr = s.noise_value
        generate_new = False
        if new_snr != im.signal_to_noise:
            generate_new = True
            im.signal_to_noise = new_snr
        im.add_noise(im.kspacedata, new_snr, im.noise_map, generate_new)

    # # 02 - Spikes
    # im.apply_spikes(im.kspacedata, im.spikes)
    #
    # # 03 - Patches
    # im.apply_patches(im.kspacedata, im.patches)
    #
    # # 04 - Reduced scan percentage
    if 'partial_fourier_value' and 'scan_percentage_value' in s:
        if s.partial_fourier_value == 100:
            v_ = s.scan_percentage_value
            im.reduced_scan_percentage(im.kspacedata, v_)
        else:
            if 'scan_percentage' in s:
                s.scan_percentage.enable = False

    # # 05 - Partial fourier
    if 'scan_percentage_value' in s:
        if s.scan_percentage_value == 100:
            v_ = s.partial_fourier_value
            zf = s.zero_fill_value
            im.partial_fourier(im.kspacedata, v_, zf)

    # 06 - High pass filter
    if 'high_pass_value' in s:
        v_ = s.high_pass_value
        im.high_pass_filter(im.kspacedata, v_)

    # # 07 - Low pass filter
    if 'low_pass_value' in s:
        v_ = s.low_pass_value
        im.low_pass_filter(im.kspacedata, v_)
    #
    # # 08 - Undersample k-space
    if 'undersample_value' in s:
        v_ = s.undersample_value
        if int(v_):
            compress = False
            if 'compress_value' in s:
                compress = s.compress_value
            im.undersample(im.kspacedata, int(v_), compress)
    #
    # # 09 - DC signal decrease
    # v_ = self.ui_decrease_dc.property("value")
    # if int(v_) > 1:
    #     im.decrease_dc(im.kspacedata, int(v_))
    #
    # # 10 - Hamming filter
    # if self.ui_hamming.property("checked"):
    #     im.hamming(im.kspacedata)
    #
    # # 11 - Acquisition simulation progress
    # if self.ui_filling.property("value") < 100:
    #     mode = self.ui_filling_mode.property("currentIndex")
    #     im.filling(im.kspacedata, self.ui_filling.property("value"), mode)

    # Get the resulting image
    im.np_ifft(kspace=im.kspacedata, out=im.img)

    # # Get display properties
    if 'k_scaling_value' in s:
        kspace_const = s.k_scaling_value
    else:
        kspace_const = -3
    # # Window values
    if 'window_width' in s:
        ww = s.window_width
    else:
        ww = 1
    if 'window_center' in s:
        wc = s.window_center
    else:
        wc = 0.5
    win_val = {'ww': ww, 'wc': wc}
    im.prepare_displays(kspace_const, win_val)


if __name__ == "__main__":
    state = st.session_state
    default_file = 'images/default.dcm'

    if 'uploaded_image' in state and state.uploaded_image is not None:
        file_to_open = state.uploaded_image
    else:
        file_to_open = default_file

    try:
        im = ImageManipulators(open_file(file_to_open))
    except Exception as err:
        st.exception(err)
        im = ImageManipulators(open_file(default_file))

    image_change(state)
    img_box, kspace_box = st.columns(2)

    if 'scan_percentage_disabled' not in state:
        state.scan_percentage_disabled = False
    if 'partial_fourier_disabled' not in state:
        state.partial_fourier_disabled = False
    # Set partial fourier and scan percentage disable each other when != 100
    if 'partial_fourier_value' in state:
        if state.partial_fourier_value != 100:
            state.scan_percentage_disabled = True
        else:
            state.scan_percentage_disabled = False
    if 'scan_percentage_value' in state:
        if state.scan_percentage_value != 100:
            state.partial_fourier_disabled = True
        else:
            state.partial_fourier_disabled = False

    # Sidebar elements
    st.sidebar.header('⚕️ K-space Explorer Online')
    st.sidebar.write('[https://kspace.app](https://k-space.app/)')
    with st.sidebar.expander("View custom image"):
        uploader = st.file_uploader(
            'Upload a file',
            key='uploaded_image',
        )

    with st.sidebar.expander("Modify K-space"):
        st.write('----------')
        partial_fourier = st.slider(
            'Partial Fourier',
            min_value=0, max_value=100, value=100,
            key='partial_fourier_value',
            disabled=state.partial_fourier_disabled)

        zero_fill = st.checkbox(
            'Zero-Fill',
            value=True,
            key='zero_fill_value')

        st.write('----------')

        noise = st.slider(
            'Signal to Noise (dB)',
            min_value=-30, max_value=30, value=30,
            key='noise_value')

        st.write('----------')

        scan_percentage = st.slider(
            'Scan Percentage',
            min_value=0, max_value=100, value=100,
            key='scan_percentage_value',
            disabled=state.scan_percentage_disabled)

        st.write('----------')

        high_pass_filter = st.slider(
            'High Pass Filter',
            min_value=0, max_value=100, value=0,
            key='high_pass_value')

        st.write('----------')

        low_pass_filter = st.slider(
            'Low Pass Filter',
            min_value=0, max_value=100, value=100,
            key='low_pass_value')

        st.write('----------')

        undersample = st.slider(
            'Undersample k-space',
            min_value=1, max_value=16, value=1,
            key='undersample_value')

        compress_check = st.checkbox(
            'Compress undersampled k-space',
            key='compress_value')

        st.write('----------')

        k_scaling = st.slider(
            'K_space scaling constant (10ⁿ)',
            min_value=-10, max_value=10, value=-3,
            key='k_scaling_value')

    with st.sidebar.expander("Image windowing tools"):
        st.slider(
            'Window width',
            min_value=0.01, max_value=1., value=1., step=0.01,
            key='window_width')

        st.slider(
            'Window center',
            min_value=0.01, max_value=1., value=0.5, step=0.01,
            key='window_center')

    st.sidebar.caption('Created by Gergely Biro')
    st.sidebar.caption('Please consider a [small donation ☕](https://www.paypal.com/paypalme/birogeri/5gbp) if you find this app useful.')

    img_box.image(im.image_display_data, use_column_width="always")
    kspace_box.image(im.kspace_display_data, use_column_width="always")
