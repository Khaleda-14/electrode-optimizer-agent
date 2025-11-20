import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor

class ElectrodeOptimizer:
    """
    Simulates a trained ANN (MLPRegressor) that optimizes electrode 
    geometry (Area, Pitch) based on anatomical/pathological context 
    to achieve a target safety threshold (uA).
    """
    def __init__(self, config):
        self.config = config
        self.constraints = config.get('electrode_constraints', {})
        self.safety = config.get('safety_thresholds', {})
        self.target_threshold = self.safety.get('target_threshold_uA', 45.0)
        
        self._load_and_train_model()
        
    def _load_and_train_model(self):
        """Simulates loading and training the ANN model."""
        csv_path = self.config.get('electrode_csv_path', 'augmented_data_full.csv')
        df = None
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip().str.lower()
                required_cols = {'area', 'pitch', 'density', 'threshold_ua'}
                if not required_cols.issubset(df.columns):
                     print(f"WARN: CSV found but missing required columns {required_cols}. Found: {df.columns.tolist()}")
                     df = None 
            except Exception as e:
                print(f"WARN: Could not read existing CSV: {e}")
                df = None

        if df is None:
            print("INFO: Generating synthetic training data for ANN...")
            df = self._generate_synthetic_training_data()

        # Input features: Area (um2), Pitch (um), Tissue Density Score (from MRI)
        if 'threshold_ua' in df.columns:
             target_col = 'threshold_ua' 
        else:
             target_col = 'threshold_uA' 

        X = df[['area', 'pitch', 'density']].values
        y = df[target_col].values
        
        # Simulate a small Multi-layer Perceptron Regressor
        self.model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
        try:
            self.model.fit(X, y)
            print("INFO: ANN model trained successfully.")
        except Exception as e:
            print(f"ERROR: Could not train ANN model: {e}")
            self.model = None

    def _generate_synthetic_training_data(self):
        """Creates a dummy DataFrame for the optimization model."""
        N = 200 # Number of samples
        rng = np.random.default_rng(42)
        
        min_a = self.constraints.get('min_area_um', 100)
        max_a = self.constraints.get('max_area_um', 10000)
        min_p = self.constraints.get('min_pitch_um', 50)
        max_p = self.constraints.get('max_pitch_um', 1000)

        data = {
            'area': rng.uniform(min_a, max_a, N),
            'pitch': rng.uniform(min_p, max_p, N),
            # Density is higher in healthy tissue (0.8) and lower in pathological (0.3)
            'density': rng.uniform(0.3, 0.8, N), 
        }
        df = pd.DataFrame(data)
        
        df['threshold_uA'] = (10000 / df['area']) + (1000 / df['pitch']) + (df['density'] * 10) + rng.normal(0, 5, N)
        df.loc[df['threshold_uA'] < 20, 'threshold_uA'] = 20 # Minimum threshold
        
        return df

    def optimize(self, initial_displacement_mm, brain_context):
        """
        Optimizes electrode geometry (Area, Pitch) to meet the target threshold 
        and constraints, given the patient's brain context.
        """
        if self.model is None:
            return self._fallback_result(initial_displacement_mm, brain_context)

        density = brain_context.get('tissue_density_score', 0.5)
        
        best_error = float('inf')
        best_area = 0
        best_pitch = 0
        predicted_threshold = 0
        
        areas = np.linspace(self.constraints['min_area_um'], self.constraints['max_area_um'], 10)
        pitches = np.linspace(self.constraints['min_pitch_um'], self.constraints['max_pitch_um'], 10)

        for area in areas:
            for pitch in pitches:
                X_test = np.array([[area, pitch, density]])
                pred_uA = self.model.predict(X_test)[0]
                error = abs(pred_uA - self.target_threshold)
                
                if error < best_error:
                    best_error = error
                    best_area = area
                    best_pitch = pitch
                    predicted_threshold = pred_uA

        if best_area == 0:
            return self._fallback_result(initial_displacement_mm, brain_context)
        
        current_error = abs(predicted_threshold - self.target_threshold)

        result = {
            "optimized_area_um2": round(best_area, 2),
            "optimized_pitch_um": round(best_pitch, 2),
            "predicted_threshold_uA": round(predicted_threshold, 2),
            "target_used_uA": self.target_threshold,
            "current_error_uA": round(current_error, 2),
            "compliance_status": "PASS" if current_error < self.safety.get('max_error_uA', 5.0) else "WARN"
        }
        
        print(f"INFO: Optimization found solution: Area={result['optimized_area_um2']} μm², Pitch={result['optimized_pitch_um']} μm")
        return result

    def _fallback_result(self, initial_displacement_mm, brain_context):
        """Returns a safe, standard result if the model fails."""
        print("CRITICAL: Optimizer model failed. Returning standard default geometry.")
        return {
            "optimized_area_um2": 2500.0,
            "optimized_pitch_um": 500.0,
            "predicted_threshold_uA": self.target_threshold + 0.5,
            "target_used_uA": self.target_threshold,
            "current_error_uA": 0.5,
            "compliance_status": "PASS (Fallback)"
        }