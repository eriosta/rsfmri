import os
import numpy as np
import itertools
import scipy.linalg as linalg


class FSLNets:
    def __init__(self, datadir, subfile, goodics):
        self.datadir = datadir
        self.subfile = subfile
        self.goodics = goodics

    def normalise_data(self, dat):
        """Demans and divides by std.
        Data is timepoints X components."""
        tmp = (dat - dat.mean(0))
        tmp = tmp / tmp.std(ddof=1)
        return tmp

    def parse_idx(self, good, n):
        """Given a list of good indicies (good), out of
        n indicies, return an array of good_idx and bad_idx."""
        all_idx = np.arange(n)
        bad_idx = np.array([x for x in all_idx if not x in good])
        good_idx = np.array([x for x in all_idx if x in good])
        test_idx = sorted(np.concatenate((good_idx, bad_idx)))
        if not all([x in good_idx for x in good]):
            raise IOError('index issue %s not in range %s' % (good, all_idx))
        return good_idx, bad_idx

    def simple_regress(self, good, bad):
        """Simple linear regression to regress bad components
        from good components."""
        result = good - np.dot(bad, np.dot(np.linalg.pinv(bad), good))
        return result

    def remove_regress_bad_components(self, dat, good):
        """Returns timepoints by good components for this subject."""
        ntimepoints, ncomponents = dat.shape
        good_ids, bad_ids = self.parse_idx(good, ncomponents)
        good_dat = dat[:, good_ids]
        bad_dat = dat[:, bad_ids]
        return self.simple_regress(good_dat, bad_dat)

    def concat_subjects(self, subjects):
        """Turn a list of subject data arrays into one array."""
        nsubs = len(subjects)
        ntimepts, ncomp = subjects[0].shape
        concat = np.array(subjects)
        concat.shape = (nsubs, ntimepts, ncomp)
        return concat

    def corrcoef(self, data):
        """Calculates the corrcoef for data with structure
        (ntimepoints * nsub) X ncomponents.
        Returns array: ncomponents X ncomponents with zeros diagonal."""
        res = np.corrcoef(data.T)
        np.fill_diagonal(res, 0)
        return res

    def partial_corr(self, data):
        """Calculates partial correlation for data with structure
        (ntimepoints X ncomponents).
        Returns array: ncomponents X ncomponents with zeros diagonal."""
        timepts, ncomp = data.shape
        all_pcorr = np.zeros((ncomp, ncomp))
        allcond = set(np.arange(ncomp))
        for a, b in itertools.combinations(allcond, 2):
            xy = data[:, np.array([a, b])]
            rest = allcond - set([a, b])
            confounds = data[:, np.array([x for x in rest])]
            part_corr = self._cond_partial_cor(xy, confounds)
            all_pcorr[a, b] = part_corr
            all_pcorr[b, a] = part_corr
        return all_pcorr

    def _cond_partial_cor(self, xy, confounds=[]):
        """Returns the partial correlation of y and x, conditioning on confounds."""
        if len(confounds):
            res = linalg.lstsq(confounds, xy)
            xy = xy - np.dot(confounds, res[0])
        return np.dot(xy[:, 0], xy[:, 1]) / np.sqrt(np.dot(xy[:, 1], xy[:, 1]) * np.dot(xy[:, 0], xy[:, 0]))

    def calc_arone(self, sdata):
        """Quick estimate of median AR(1) coefficient
        across subjects concatenated data: nsub, ntimepoints X ncomponents"""
        arone = np.sum(sdata[:, :, :-1] * sdata[:, :, 1:], 2) / np.sum(sdata * sdata, 2)
        return np.median(arone)

    def _calc_r2z_correction(self, sdata, arone):
        """Uses the precomputed median auto-regressive AR(1) coefficient
        to z-transform subjects' data."""
        nsub, ntimepts, nnodes = sdata.shape
        null = np.zeros(sdata.shape)
        null[:, 0, :] = np.random.randn(nsub, nnodes)
        for i in range(ntimepts - 1):
            null[:, i + 1, :] = null[:, i, :] * arone
        null[:, 1:, :] = null[:, 1:, :] + np.random.randn(nsub, ntimepts - 1, nnodes)
        non_diag = np.empty((nsub, nnodes * nnodes - nnodes))
        for sub, slice in enumerate(null):
            tmpr = np.corrcoef(slice)
            non_diag[sub] = tmpr[np.eye(nnodes) < 1]
        tmpz = 0.5 * np.log((1 + non_diag) / (1 - non_diag))
        r_to_z_correct = 1.0 / tmpz.std()
        return r_to_z_correct

    def r_to_z(self, subs_node_stat, sdata):
        """Calculates and returns z-transformed data."""
        arone = self.calc_arone(sdata)
        r_to_z_val = self._calc_r2z_correction(sdata, arone)
        zdat = 0.5 * np.log((1 + subs_node_stat) / (1 - subs_node_stat)) * r_to_z_val
        return zdat

    def load_data(self):
        """Loads and preprocesses the data."""
        with open(self.subfile, 'r') as f:
            sublist = f.read().splitlines()
        group_ts = {}  # Dictionary to hold timeseries of all subjects
        group_stats = []  # List to hold correlation matrices of all subjects
        for subj in sublist:
            infile = '_'.join(['dr_stage1', subj, 'and_confound_regressors_6mm'])
            data = np.loadtxt(os.path.join(self.datadir, infile), dtype='float')
            norm_data = self.normalise_data(data)  # Load data and normalize
            clean_data = self.remove_regress_bad_components(norm_data, self.goodics)
            nnodes = clean_data.shape[1]
            node_stat = self.corrcoef(clean_data)  # Calculate correlation matrix of all good components
            reshaped_stat = node_stat.reshape((1, nnodes * nnodes))
            group_ts[subj] = clean_data  # Append subject data to group data
            group_stats.append(reshaped_stat)
        self.group_ts = group_ts
        self.group_stats = group_stats

    def preprocess_data(self):
        """Preprocesses the data."""
        concat_ts = self.concat_subjects(self.group_ts.values())  # Create concatenated matrix of subjects' timeseries data
        nsubs, ntimepts, nnodes = concat_ts.shape
        concat_stats = np.array(self.group_stats)  # Create concatenated matrix of subjects' correlation matrices
        reshaped_stats = concat_stats.reshape((nsubs, nnodes * nnodes))
        self.concat_ts = concat_ts
        self.concat_stats = reshaped_stats

    def run_tests(self):
        """Runs the test cases."""
        from unittest import TestCase
        from numpy.testing import assert_raises, assert_equal, assert_almost_equal

        class TestFSLNets(TestCase):

            def setUp(self):
                """Create small example data."""
                self.data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
                self.good = [0, 1, 2]

            def test_normalize_data(self):
                cleaned = self.normalise_data(self.data)
                assert_equal(cleaned.shape, self.data.shape)
                expected = np.array([[-1.41421356, -1.41421356, -1.41421356],
                                     [-0.70710678, -0.70710678, -0.70710678],
                                     [0., 0., 0.],
                                     [0.70710678, 0.70710678, 0.70710678],
                                     [1.41421356, 1.41421356, 1.41421356]])
                assert_almost_equal(cleaned, expected)

            def test_parse_idx(self):
                good = self.good
                ntimepts, ncomp = self.data.shape
                good_idx, bad_idx = self.parse_idx(good, ncomp)
                assert_equal(good_idx, np.array([0, 1, 2]))
                assert_equal(bad_idx, np.array([]))

            def test_regress(self):
                ntimepts, ncomp = self.data.shape
                cleaned = self.normalise_data(self.data)
                gids, bids = self.parse_idx(self.good, ncomp)
                good = cleaned[:, gids]
                bad = cleaned[:, bids]
                res = self.simple_regress(good, bad)
                expected = np.array([[-0.50709255, 0., 0.50709255],
                                     [-0.50709255, 0., 0.50709255],
                                     [-0.50709255, 0., 0.50709255],
                                     [-0.50709255, 0., 0.50709255],
                                     [-0.50709255, 0., 0.50709255]])
                assert_almost_equal(res, expected)

            def test_remove_regress_bad_components(self):
                cleaned = self.normalise_data(self.data)
                res = self.remove_regress_bad_components(cleaned, self.good)
                expected = np.array([[-0.50709255, 0., 0.50709255],
                                     [-0.50709255, 0., 0.50709255],
                                     [-0.50709255, 0., 0.50709255],
                                     [-0.50709255, 0., 0.50709255],
                                     [-0.50709255, 0., 0.50709255]])
                assert_almost_equal(res, expected)

            def test_concat_subjects(self):
                tmp = [self.data, self.data]
                ntimepts, ncomp = self.data.shape
                concat = self.concat_subjects(tmp)
                assert_equal(concat.shape, (2, ntimepts, ncomp))
                assert_equal(concat[0, :, :], self.data)

            def test_corrcoef(self):
                res = self.corrcoef(self.data)
                _, ncomp = self.data.shape
                assert_equal(res.shape, (ncomp, ncomp))
                assert_equal(np.diag(res), np.zeros(ncomp))

            def test_calc_arone(self):
                sdata = self.concat_subjects([self.data, self.data])
                arone = self.calc_arone(sdata)
                assert_almost_equal(0.49666667, arone)

        suite = TestFSLNets()
        suite.setUp()
        suite.test_normalize_data()
        suite.test_parse_idx()
        suite.test_regress()
        suite.test_remove_regress_bad_components()
        suite.test_concat_subjects()
        suite.test_corrcoef()
        suite.test_calc_arone()

    def run(self):
        """Runs the FSLNets analysis."""
        self.load_data()
        self.preprocess_data()
        self.run_tests()


if __name__ == '__main__':
    datadir = '/home/jagust/rsfmri_ica/data/OldICA_IC0_ecat_2mm_6fwhm_125.gica/dual_regress'
    subfile = '/home/jagust/rsfmri_ica/Spreadsheets/Filelists/OLDICA_5mm_125_orig_sublist.txt'
    goodics = [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 17, 20, 22, 25, 30, 33]
    fsln = FSLNets(datadir, subfile, goodics)
    fsln.run()
