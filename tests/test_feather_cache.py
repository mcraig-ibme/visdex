import pandas as pd

from visdex.data.feather_cache import FeatherCache

COLS = [
    "subjid",
    "sessid",
    "meas1",
    "meas2",
    "strmeas",
]
DATA = [
    ['abc1', 'mri1', 1.3, 48, 'jkl'], 
    ['abc2', 'mri2', 1.8, 17, 'qwe'], 
    ['abc3', 'mri2', 1.2, 12, 'dfg'], 
    ['abc4', 'mri3', 0.4, 28, 'zxc'], 
]

class TestFeatherCache:

    def setup_method(self, _test_method):
        self._cache = FeatherCache()

    def test_no_store(self):
        """ Unknown df still loads as empty DF """
        df2 = self._cache.load("missing")
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == 0
        assert len(df2.columns) == 0

    def test_empty_df(self):
        """ Empty data frame loads/stores OK """
        df = pd.DataFrame()
        self._cache.store("empty", df)
        df2 = self._cache.load("empty")
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == 0
        assert len(df2.columns) == 0

    def test_df_none(self):
        """ None is treated as empty data frame """
        self._cache.store("empty", None)
        df2 = self._cache.load("empty")
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == 0
        assert len(df2.columns) == 0

    def test_df_default_index(self):
        """ Test DF with default index """
        df = pd.DataFrame(DATA, columns=COLS)
        self._cache.store("test", df)
        df2 = self._cache.load("test")
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == len(DATA)
        assert list(df2.columns) == COLS
        for idx, row in df2.iterrows():
            assert(list(row) == DATA[idx])

    def test_df_default_index_keep(self):
        """ Test restoring a DF with default index and keep index cols does *not* add the index as a column"""
        df = pd.DataFrame(DATA, columns=COLS)
        self._cache.store("test", df)
        df2 = self._cache.load("test", keep_index_cols=True)
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == len(DATA)
        assert list(df2.columns) == COLS
        for idx, row in df2.iterrows():
            assert(list(row) == DATA[idx])

    def test_df_single_index(self):
        """ Test DF with single-column index """
        df = pd.DataFrame(DATA, columns=COLS)
        df.set_index('subjid', drop=True, inplace=True, verify_integrity=False)
        assert list(df.columns) == COLS[1:]
        assert(list(df.index.names) == ['subjid'])

        self._cache.store("test", df)
        df2 = self._cache.load("test")

        assert isinstance(df2, pd.DataFrame)
        assert list(df2.columns) == COLS[1:]
        assert(list(df2.index.names) == ['subjid'])
        assert len(df2) == len(DATA)
        for idx, row in enumerate(df2.iterrows()):
            subjid, rowdata = row
            assert(subjid) == DATA[idx][0]
            assert(list(rowdata) == DATA[idx][1:])

    def test_df_single_index_keep(self):
        """ Test loading a DF with single-column index and keeping the index columns """
        df = pd.DataFrame(DATA, columns=COLS)
        df.set_index('subjid', drop=True, inplace=True, verify_integrity=False)
        assert list(df.columns) == COLS[1:]
        assert(list(df.index.names) == ['subjid'])

        self._cache.store("test", df)
        df2 = self._cache.load("test", keep_index_cols=True)

        assert isinstance(df2, pd.DataFrame)
        assert list(df2.columns) == COLS
        assert(list(df2.index.names) == ['subjid'])
        assert len(df2) == len(DATA)
        for idx, row in enumerate(df2.iterrows()):
            subjid, rowdata = row
            assert(subjid) == DATA[idx][0]
            assert(list(rowdata) == DATA[idx])

    def test_df_multi_index(self):
        """ Test DF with multi-column index """
        df = pd.DataFrame(DATA, columns=COLS)
        df.set_index(['subjid', 'sessid'], drop=True, inplace=True, verify_integrity=False)
        assert list(df.columns) == COLS[2:]
        assert(list(df.index.names) == ['subjid', 'sessid'])

        self._cache.store("test", df)
        df2 = self._cache.load("test")

        assert isinstance(df2, pd.DataFrame)
        assert list(df2.columns) == COLS[2:]
        assert(list(df2.index.names) == ['subjid', 'sessid'])
        assert len(df2) == len(DATA)
        for idx, row in enumerate(df2.iterrows()):
            index, rowdata = row
            assert(list(index)) == DATA[idx][:2]
            assert(list(rowdata) == DATA[idx][2:])

    def test_df_multi_index_keep(self):
        """ Test restoring DF with multi-column index and keeping the index columns """
        df = pd.DataFrame(DATA, columns=COLS)
        df.set_index(['subjid', 'sessid'], drop=True, inplace=True, verify_integrity=False)
        assert list(df.columns) == COLS[2:]
        assert(list(df.index.names) == ['subjid', 'sessid'])

        self._cache.store("test", df)
        df2 = self._cache.load("test", keep_index_cols=True)

        assert isinstance(df2, pd.DataFrame)
        assert list(df2.columns) == COLS
        assert(list(df2.index.names) == ['subjid', 'sessid'])
        assert len(df2) == len(DATA)
        for idx, row in enumerate(df2.iterrows()):
            index, rowdata = row
            assert(list(index)) == DATA[idx][:2]
            assert(list(rowdata) == DATA[idx])



