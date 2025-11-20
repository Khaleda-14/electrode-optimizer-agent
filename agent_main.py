import json
import time
import os
import numpy as np
import sys

try:
    from mri_processor import MRIProcessor
    from multimodal_analyzer import MultimodalAnalyzer
    from electrode_optimizer import ElectrodeOptimizer
except ImportError as e:
    print(f"CRITICAL: Missing component file: {e}. Ensure all .py files are present.")
    sys.exit(1)


class AutonomousElectrodeAgent:
    """
    Orchestrates the entire pipeline: Data -> AI Analysis -> Optimization -> Report.
    """
    def __init__(self, config_path="config.json"):
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print(f"WARN: Config file '{config_path}' not found. Using hardcoded defaults.")
            self.config = {
                 "output_dir": "results", 
                 "pretrained_model_path": "pretrained_ann.h5",
                 "electrode_csv_path": "augmented_data_full.csv",
                 "subject_mri_path": None,
                 "default_displacement_mm": 120.0,
                 "electrode_constraints": {
                    "min_pitch_um": 50.0, "max_pitch_um": 1000.0,
                    "min_area_um": 100.0, "max_area_um": 10000.0
                 },
                 "safety_thresholds": {"target_threshold_uA": 45.0, "max_error_uA": 5.0}
            }

        os.makedirs(self.config.get('output_dir', 'results'), exist_ok=True)
        
        self.mri_proc = MRIProcessor(self.config)
        self.analyzer = MultimodalAnalyzer(self.config)
        self.optimizer = ElectrodeOptimizer(self.config)

    def run_autonomous_analysis(self, patient_id, displacement_mm=None, mri_image_path=None):
        """
        Runs the full analysis pipeline using configuration defaults or user inputs.
        Returns the full report dictionary, including NumPy arrays for visualization.
        """
        print(f"\n=== STARTING AUTONOMOUS AGENT: {patient_id} ===")
        start_time = time.time()
        
        mri_path = mri_image_path if mri_image_path else self.config.get('subject_mri_path', None)
        disp_mm = displacement_mm if displacement_mm is not None else self.config.get('default_displacement_mm', 120.0)
        mri_data = self.mri_proc.load_and_process(mri_path)
        if mri_data is None or mri_data.get('full_volume') is None:
            print("CRITICAL ERROR: MRI Processing failed or returned empty data.")
            return None

        brain_context = self.analyzer.analyze(mri_data)
        geometry_result = self.optimizer.optimize(disp_mm, brain_context)
        json_mri_data = {
            "full_volume": mri_data['full_volume'].tolist() if isinstance(mri_data['full_volume'], np.ndarray) else mri_data['full_volume'],
            "segmentation_mask": mri_data['segmentation_mask'].tolist() if isinstance(mri_data['segmentation_mask'], np.ndarray) else mri_data['segmentation_mask']
        }
        
        final_report = {
            "patient_id": patient_id,
            "timestamp": time.ctime(),
            "mri_source": mri_data.get('modality', 'Unknown'),
            "mri_data_json_friendly": json_mri_data,
            "brain_analysis": brain_context,
            "final_recommendations": geometry_result,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
        self._save_results(final_report, patient_id)
        final_report['mri_data'] = {
            "full_volume": mri_data['full_volume'],
            "segmentation_mask": mri_data['segmentation_mask']
        }
        
        return final_report

    def _save_results(self, report, patient_id):
        """Saves the final report to the results directory."""
        output_dir = self.config.get('output_dir', 'results')
        filepath = os.path.join(output_dir, f"analysis_{patient_id}_{int(time.time())}.json")
        report_to_save = report.copy()
        report_to_save.pop('mri_data', None) 
        report_to_save['mri_data'] = report['mri_data_json_friendly']
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report_to_save, f, indent=4)
            print(f"INFO: Analysis report saved to {filepath}")
        except Exception as e:
            print(f"ERROR: Could not save report to file: {e}")

if __name__ == "__main__":
    
    agent = AutonomousElectrodeAgent()
    results = agent.run_autonomous_analysis(
        patient_id="TestSubject_CL", 
        displacement_mm=None, 
        mri_image_path=None
    )
    if results:
        print("\nOptimization Complete. Results:")
        print(json.dumps(results['final_recommendations'], indent=4))