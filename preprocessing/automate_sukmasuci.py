import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from joblib import dump
import os
from datetime import datetime


# ============================================================
# FIX LAMBDA ERROR → Fungsi harus global (bisa di-pickle)
# ============================================================
def safe_log_transform(x):
    """Safe log transform: mencegah error untuk nilai negatif."""
    return np.log1p(np.maximum(x, 0))


class DataPreprocessor:
    """
    Class untuk handle preprocessing data pipa secara otomatis.
    """

    def __init__(self, dataset_path, target_col='Condition'):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.data = None
        self.label_encoder = None
        self.preprocessor = None
        self.feature_names = []

    def load_data(self):
        """Load dataset dari file CSV"""
        self.data = pd.read_csv(self.dataset_path)
        print(f"Data berhasil dimuat: {self.data.shape[0]} baris, {self.data.shape[1]} kolom")
        return self

    def prepare_features(self):
        """Pisahkan fitur dan tentukan fitur skewed / normal / kategoris"""
        if 'Pipe_Size_mm' in self.data.columns:
            self.data = self.data.drop(columns=['Pipe_Size_mm'])
            print("Kolom Pipe_Size_mm sudah dibuang")

        self.X = self.data.drop(columns=[self.target_col])
        self.y = self.data[self.target_col]

        numeric_cols = self.X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()

        self.skewed_cols = [col for col in ['Thickness_mm', 'Material_Loss_Percent']
                            if col in numeric_cols]

        self.normal_cols = [col for col in numeric_cols if col not in self.skewed_cols]

        print(f"Fitur skewed: {self.skewed_cols}")
        print(f"Fitur normal: {self.normal_cols}")
        print(f"Fitur kategoris: {self.categorical_cols}")

        return self

    def split_data(self, test_size=0.2, random_state=42):
        """Split data train-test"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        print(f"Data train: {self.X_train.shape[0]} | Data test: {self.X_test.shape[0]}")
        return self

    def encode_target(self):
        """Encode label target"""
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)

        print("Target di-encode:",
              dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))))
        return self

    def create_pipeline(self):
        """Buat pipeline preprocessing"""

        skewed_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('log_transform', FunctionTransformer(safe_log_transform)),
            ('scaler', MinMaxScaler())
        ])

        normal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer([
            ('skewed_features', skewed_pipeline, self.skewed_cols),
            ('normal_features', normal_pipeline, self.normal_cols),
            ('categorical_features', categorical_pipeline, self.categorical_cols)
        ])

        print("Pipeline preprocessing sudah dibuat")
        return self

    def fit_transform_data(self):
        """Fit dan transform data"""
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)

        self._extract_feature_names()

        print(f"Preprocessing selesai! Shape akhir: {self.X_train_processed.shape}")
        return self

    def _extract_feature_names(self):
        """Ambil nama fitur setelah one-hot encoding"""
        self.feature_names = []
        self.feature_names.extend(self.skewed_cols)
        self.feature_names.extend(self.normal_cols)

        cat_transformer = self.preprocessor.named_transformers_['categorical_features']
        encoder = cat_transformer.named_steps['encoder']
        cat_feature_names = encoder.get_feature_names_out(self.categorical_cols)

        self.feature_names.extend(cat_feature_names)

    def save_results(self, output_dir):
        """Simpan hasil preprocessing"""
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(self.X_train_processed, columns=self.feature_names).to_csv(
            os.path.join(output_dir, 'X_train.csv'), index=False
        )
        pd.DataFrame(self.X_test_processed, columns=self.feature_names).to_csv(
            os.path.join(output_dir, 'X_test.csv'), index=False
        )
        pd.DataFrame(self.y_train_encoded, columns=[self.target_col]).to_csv(
            os.path.join(output_dir, 'y_train.csv'), index=False
        )
        pd.DataFrame(self.y_test_encoded, columns=[self.target_col]).to_csv(
            os.path.join(output_dir, 'y_test.csv'), index=False
        )

        pd.Series(self.feature_names).to_csv(
            os.path.join(output_dir, 'feature_names.csv'), index=False, header=False
        )

        # SIMPAN PIPELINE TANPA ERROR
        dump(self.preprocessor, os.path.join(output_dir, 'pipeline.joblib'))
        dump(self.label_encoder, os.path.join(output_dir, 'pipeline_label_encoder.joblib'))

        print(f"✓ Semua hasil preprocessing sudah tersimpan di: {output_dir}")

    def run_full_pipeline(self, output_dir):
        """Jalankan seluruh proses"""
        (self.load_data()
         .prepare_features()
         .split_data()
         .encode_target()
         .create_pipeline()
         .fit_transform_data()
         .save_results(output_dir)
         )
        return self


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    script_location = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.abspath(os.path.join(script_location, '..'))

    data_file = os.path.join(root_folder, 'dataset_raw', 'market_pipe_thickness_loss_dataset.csv')

    output_folder = os.path.join(script_location, 'preprocessed_data_auto')

    print("=" * 60)
    print("MULAI PREPROCESSING DATA PIPA")
    print("=" * 60)

    preprocessor = DataPreprocessor(data_file, target_col='Condition')
    preprocessor.run_full_pipeline(output_folder)

    print("=" * 60)
    print("PREPROCESSING SELESAI!")
    print("=" * 60)
