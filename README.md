# Autonomous AI Agent for Neuro-Electrode Geometry Optimization

A fully autonomous AI-powered system for optimizing neural electrode geometry based on patient-specific MRI connectome data.

## Features

**Interactive Desktop Application**: Modern GUI with intuitive interface for easy use  
**Fully Autonomous Analysis**: End-to-end optimization from MRI data to electrode geometry without manual intervention  
**Multimodal AI Integration**: Combines vision-language models for context-aware brain region analysis  
**Zero-Shot Generalization**: Analyzes novel MRI scans without additional fine-tuning  
**Adaptive Strategy Selection**: Automatically selects optimal analysis approach based on data characteristics  
**Real-Time Progress Tracking**: Visual progress indicators and status updates during analysis  
**Results Visualization**: Automatic display of electrode geometry plots and threshold graphs  
**Export Functionality**: Save results to JSON format for further analysis  
**Memory-Efficient**: Optimized for Google Colab with memory constraints (< 4GB RAM usage)  
**Clinical-Grade Output**: Generates comprehensive electrode geometry specifications (Area (mm²), Pitch (mm))  

## Quick Start

### Option 1: Interactive Desktop GUI (Recommended)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the GUI Application**
   ```bash
   python launch_gui.py
   ```
   Or:
   ```bash
   python gui_main.py
   ```

3. **Use the GUI**
   - Click "Browse MRI File" to select your MRI image (.nii or .nii.gz)
   - Enter patient ID and target region (optional)
   - Configure electrode parameters (displacement, target threshold)
   - Click "Start Analysis" to run the optimization
   - View results, visualizations, and export data

### Option 2: Command Line Interface

```bash
python agent_main.py --displacement-mm 120 --patient-id subject_01 --mri-image t1w.nii
```

### Option 3: Python API

```python
from agent_main import AutonomousElectrodeAgent

agent = AutonomousElectrodeAgent(config_path="config.json")

results = agent.run_autonomous_analysis(
    patient_id="subject_01",
    displacement_mm=120.0,
    mri_image_path="t1w.nii"
)

print(results['final_recommendations'])
```

### Option 4: Google Colab

1. Upload all project files to Colab
2. Install dependencies: `!pip install -r requirements.txt`
3. Run the Python API code above

## System Architecture

### Pipeline Stages

1. **Data Acquisition** (`mri_processor.py`)
   - Loads MRI connectome data from public datasets (HCP, Fiber Data Hub)
   - Preprocesses and downsamples for memory efficiency
   - Extracts structural connectivity matrices

2. **Adaptive Strategy Selection** (`agent_main.py`)
   - Analyzes data characteristics (modality, quality, resolution)
   - Automatically selects optimal analysis approach
   - Identifies target brain regions

3. **Multimodal AI Analysis** (`multimodal_analyzer.py`)
   - Zero-shot brain region identification
   - Visual feature extraction from MRI slices
   - Connectivity pattern analysis
   - Identifies optimal stimulation zones and avoidance regions

4. **Electrode Optimization** (`electrode_optimizer.py`)
   - Loads the pretrained ANN (`pretrained_ann.h5`) and dataset scaler (`augmented_data_full.csv`)
   - Uses GA to optimize Area & Pitch for the displacement supplied by the user
   - Minimizes the ANN-predicted threshold current toward 45 µA (0.045 mA)
   - Generates a 2D square anode/cathode visualization with the optimized pitch spacing

5. **Validation & Refinement** (`electrode_optimizer.py`)
   - Safety constraint verification
   - Therapeutic window calculation
   - Automatic refinement if constraints violated

### Output Files

- `results/analysis_<patient_id>.json` - Complete analysis report
- `results/geometry_<patient_id>.csv` - Optimized electrode specifications

## Pretrained Model & Dataset Format

- `pretrained_ann.h5`  
  - Input: `[Displacement(mm), Area(mm), Pitch(mm)]`  
  - Output: `Threshold Current (mA)`  
  - Format: TensorFlow/Keras `.h5` or `.keras`

- `augmented_data_full.csv`  
  ```
  Displacement(mm), Area(mm), Pitch(mm), Threshold-Current(mA)
  ```
  These bounds are ingested automatically to keep GA searches inside the ANN's training regime. The GA objective converts the ANN output to microamps and minimizes the absolute error to the target 45 µA.

## MRI Connectome Data Sources

If `subject_mri_path` (for example `t1w.nii`) is configured, the agent loads that file, derives a surrogate connectome, and extracts representative slices for the multimodal analyzer. When no file is available it falls back to simulated HCP-style data. Additional public sources:

1. **Human Connectome Project (HCP)**
   - URL: https://db.humanconnectome.org
   - Free with data use agreement
   - Download structural MRI and diffusion data

2. **Fiber Data Hub**
   - URL: https://brain.labsolver.org
   - 50,000+ preprocessed scans
   - Direct integration with DSI Studio

3. **MICA-MICs Dataset**
   - 50 healthy adults with multimodal MRI
   - BIDS-compliant format
   - Includes connectivity matrices

### Loading Custom MRI Data

Set `subject_mri_path` inside `config.json` to any `.nii` or `.nii.gz` file. The processor will:

1. Load the subject volume via nibabel
2. Normalize intensities and downsample to the requested resolution
3. Derive a lightweight 256×256 connectome directly from the MRI intensity distribution
4. Tag downstream reports with the runtime `patient_id`

If you omit the setting the agent simply reverts to the simulated HCP sample.

## Configuration

Edit `config.json` to customize:

```json
{
  "memory_limit_mb": 4096,
  "cache_dir": "mri_cache",
  "output_dir": "results",
  "pretrained_model_path": "pretrained_ann.h5",
  "electrode_csv_path": "augmented_data_full.csv",
  "subject_mri_path": "t1w.nii",
  "default_displacement_mm": 100.0,
  "target_mri_resolution": 2.0,
  "n_representative_slices": 5,
  "electrode_constraints": {
    "min_pitch_mm": 0.0,
    "max_pitch_mm": 1000.0,
    "min_area_mm": 0.0,
    "max_area_mm": 2000.0
  },
  "safety_thresholds": {
    "target_threshold_uA": 45.0,
    "max_error_uA": 5.0
  }
}
```

## Example Output

```json
{
  "optimal_electrode_geometry": {
    "Area_um^2:100
    "pitch_um":300 ,
    
  },
  "stimulation_parameters": {
    "recommended_amplitude_ma": 2.3,
    "therapeutic_window_ma": 1.8,
    "pulse_width_us": 60,
    "frequency_hz": 130
  },
  "target_coordinates": {
    "mni_coordinates": [12, -15, -5],
    "target_region": "subthalamic_nucleus",
    "confidence": 0.87
  }
}
```

## Memory Optimization Tips for Colab

1. **Process slices sequentially** instead of loading full 3D volumes
2. **Use float32** instead of float64 for MRI data
3. **Enable quantization** for multimodal models (`"use_quantization": true`)
4. **Call `gc.collect()`** between pipeline stages
5. **Downsample MRI** to 2mm resolution (configurable)

## Troubleshooting

### Out of Memory Error
- Reduce `target_mri_resolution` to 3.0 or 4.0
- Decrease `n_representative_slices` to 3
- Set `memory_limit_mb` to 2048

### Model Not Found
- Ensure `pretrained_ann.h5` is in the same directory
- Check file path in `config.json`

### CSV Data Not Found
- Create sample data: agent will auto-generate placeholder data
- Ensure `augmented_data_full.csv` is uploaded

## Citation

If you use this system in your research, please cite:

```bibtex
@software{autonomous_electrode_agent,
  title = {Autonomous AI Agent for Neuro-Electrode Geometry Optimization},
  year = {2025},
  note = {Neuro stimulation electrode optimization using multimodal AI}
}
```

## References

**MRI Connectome Datasets:**
- Human Connectome Project: https://www.humanconnectome.org
- Fiber Data Hub (2025): https://brain.labsolver.org
- MICA-MICs Dataset: https://github.com/MICA-MNI/micapipe

**Electrode Optimization:**
- Steigerwald et al. (2018): "Directional Deep Brain Stimulation"
- Butson & McIntyre (2008): "Current steering to control DBS"

**Multimodal AI:**
- LLaVA-Med: https://github.com/microsoft/LLaVA-Med
- BiomedCLIP for medical imaging zero-shot learning

## License

Research use only. Not approved for clinical use.

## Support

For issues or questions, please check:
1. Configuration in `config.json`
2. Memory usage logs
3. Error messages in console output
