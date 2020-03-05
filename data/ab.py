from .data import LinearRandomDataset


class LinearRandomAirbnbDataset(LinearRandomDataset):
    def __init__(self, file_path, sigma=0, seed=None):
        super().__init__(48895, 16, sigma, seed)
        self._X = _load_data(file_path)

    def _load_data(self, file_path):
        df = pd.read_csv("dataset/AB_NYC_2019.csv", index_col=None)
        # One hot encoding
        oe = OneHotEncoder()
        df_categorical = oe.fit_transform(df[CATEGORICAL]).toarray()

        # Missing inputing
        df_numeric = df[NUMERIC]
        df_numeric = df_numeric.fillna(df_numeric.mean())

        # Standardize
        df_numeric = (df_numeric - df_numeric.mean()) / df_numeric.std()

        # Concatnate two parts
        df_all = np.concatenate([df_categorical, df_numeric], axis=1)
        return df_all
