import numpy as np
import pytest

from ..detect_peaks import detect_peaks

class TestDetectPeaks:
    """
    """
    def test_random_peaks(self):
        """Tests detection of peaks in randomized data.
        """
        y = np.random.randn(100)
        y[60:81] = np.nan
        # detect all peaks
        ind = detect_peaks(y, show=False)
        # Test that peaks are actually peaks
        for i in ind:
            assert (y[i] > y[i+1] and y[i] > y[i-1])


    def test_min_peak_distance(self):
        """Tests that peaks are separated by minimum peak distance.
        """
        y = np.array(([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]))
        # set minimum peak distance = 2
        mpd = 2
        peaks = detect_peaks(y, mpd=mpd, show=False)
        for idx, ind in enumerate(peaks):
            start = max(0, ind - mpd)
            end = min(len(y) - 1, ind + mpd)
            # Look for possible peaks before/after this one, inside mpd range
            peaks_before = np.array([i for i in range(start, ind) if y[i] > y[ind]])
            peaks_after = np.array([i for i in range(ind+1, end+1) if y[i] > y[ind]])

            # # remove bad peaks if within minimum peak distance of previous peak
            if idx > 0:
                ind_prev = peaks[idx - 1]
                idel = np.zeros(peaks_before.size, dtype=bool)
                for i in range(peaks_before.size):
                    if ((peaks_before[i] - ind_prev) <= mpd and
                        y[ind_prev] > y[peaks_before[i]]
                        ):
                        idel[i] = 1
                peaks_before = peaks_before[~idel]

            # remove bad peaks if within minimum peak distance of next peak
            if idx < len(peaks) - 1:
                ind_next = peaks[idx + 1]
                idel = np.zeros(peaks_after.size, dtype=bool)
                for i in range(peaks_after.size):
                    if ((ind_next - peaks_after[i]) <= mpd and
                        y[ind_next] > y[peaks_after[i]]
                        ):
                        idel[i] = 1
                peaks_after = peaks_after[~idel]

            assert (peaks_before.size == 0 and peaks_after.size == 0)


    def test_sin_peaks_min_distance(self):
        """Tests detection of peaks in randomized sin wave with min peak distance.
        """
        y = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5.
        # set minimum peak height = 0 and minimum peak distance = 20
        mpd = 20
        peaks = detect_peaks(y, mph=0, mpd=mpd, show=False)
        for idx, ind in enumerate(peaks):
            start = max(0, ind - mpd)
            end = min(len(y) - 1, ind + mpd)
            # Look for possible peaks before/after this one, inside mpd range
            peaks_before = np.array([i for i in range(start, ind) if y[i] > y[ind]])
            peaks_after = np.array([i for i in range(ind+1, end+1) if y[i] > y[ind]])

            # first and last values cannot be peaks/valleys
            if peaks_before.size and peaks_before[0] == 0:
                peaks_before = peaks_before[1:]
            if peaks_after.size and peaks_after[-1] == y.size-1:
                peaks_after = peaks_after[:-1]

            # # remove bad peaks if within minimum peak distance of previous peak
            if idx > 0:
                ind_prev = peaks[idx - 1]
                idel = np.zeros(peaks_before.size, dtype=bool)
                for i in range(peaks_before.size):
                    if ((peaks_before[i] - ind_prev) <= mpd and
                        y[ind_prev] > y[peaks_before[i]]
                        ):
                        idel[i] = 1
                peaks_before = peaks_before[~idel]

            # remove bad peaks if within minimum peak distance of next peak
            if idx < len(peaks) - 1:
                ind_next = peaks[idx + 1]
                idel = np.zeros(peaks_after.size, dtype=bool)
                for i in range(peaks_after.size):
                    if ((ind_next - peaks_after[i]) <= mpd and
                        y[ind_next] > y[peaks_after[i]]
                        ):
                        idel[i] = 1
                peaks_after = peaks_after[~idel]

            assert (peaks_before.size == 0 and peaks_after.size == 0)


    def test_valley_detect(self):
        """ Tests detection of valleys instead of peaks.
        """
        y = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5.
        # detection of valleys instead of peaks
        mpd = 20
        valleys = detect_peaks(y, mph=0, mpd=mpd, valley=True, show=False)
        for idx, ind in enumerate(valleys):
            start = max(0, ind - mpd)
            end = min(len(y) - 1, ind + mpd)
            # Look for possible valleys before/after this one, inside mpd range
            valleys_before = np.array([i for i in range(start, ind) if y[i] < y[ind]])
            valleys_after = np.array([i for i in range(ind+1, end+1) if y[i] < y[ind]])

            # first and last values cannot be peaks/valleys
            if valleys_before.size and valleys_before[0] == 0:
                valleys_before = valleys_before[1:]
            if valleys_after.size and valleys_after[-1] == y.size-1:
                valleys_after = valleys_after[:-1]

            # # remove bad valleys if within minimum peak distance of previous valley
            if idx > 0:
                ind_prev = valleys[idx - 1]
                idel = np.zeros(valleys_before.size, dtype=bool)
                for i in range(valleys_before.size):
                    if ((valleys_before[i] - ind_prev) <= mpd and
                        y[ind_prev] < y[valleys_before[i]]
                        ):
                        idel[i] = 1
                valleys_before = valleys_before[~idel]

            # remove bad valley if within minimum peak distance of next valley
            if idx < len(valleys) - 1:
                ind_next = valleys[idx + 1]
                idel = np.zeros(valleys_after.size, dtype=bool)
                for i in range(valleys_after.size):
                    if ((ind_next - valleys_after[i]) <= mpd and
                        y[ind_next] < y[valleys_after[i]]
                        ):
                        idel[i] = 1
                valleys_after = valleys_after[~idel]

            assert (valleys_before.size == 0 and valleys_after.size == 0)


    def test_edge_detection(self):
        """Checks that edges are detected for flat peaks.
        """
        y = [0, 1, 1, 0, 1, 1, 0]
        # detect both edges
        peaks = detect_peaks(y, edge='both', show=False)
        assert ([1, 2, 4, 5] in peaks)


    def test_peak_threshold(self):
        """Tests that peaks are separated by threshold value.
        """
        y = [-2, 1, -2, 2, 1, 1, 3, 0]
        # set threshold = 2
        threshold = 2
        peaks = detect_peaks(y, threshold=threshold, show=False)
        for ind in peaks:
            # Check for peaks less than threshold from their neighbors
            assert (y[ind] >= y[ind + 1] and y[ind] >= y[ind - 1])
