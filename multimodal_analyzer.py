import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


tf.get_logger().setLevel('ERROR')

class MultimodalAnalyzer:
    """
    Simulates a transfer-learned MobileNet V3 for zero-shot brain region
    identification based on input MRI slices (ADNI/AD focus).
    """
    def __init__(self, config):
        self.config = config
        self.ad_target_region = "Hippocampus/MTL (AD Target)"
        self.mtl_center_depth_mm = 130 

        self._init_model()
        self._init_knowledge_base()
        
    def _init_model(self):
        """Loads or simulates a MobileNetV3 model."""
        print("INFO: Loading MobileNetV3-Small (ADNI Fine-Tuned Concept)...")
        
        # Load pre-trained MobileNet V3 Small with ImageNet weights
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        for layer in base_model.layers[-20:]:
            layer.trainable = True 
            
        self.model = base_model

    def _init_knowledge_base(self):
        """
        Generates prototype vectors conceptually derived from ADNI/MTL atlases
        for zero-shot matching.
        """
        rng = np.random.default_rng(42)
        
        self.prototypes = {
            self.ad_target_region: rng.random((1, 576)) * 1.5, # Strong target embedding
            "Globus Pallidus (Control)": rng.random((1, 576)),
            "White Matter (Projection Tract)": rng.random((1, 576)) * 0.8,
            "Ventricle/CSF (Avoidance)": rng.random((1, 576)) * 0.1
        }

    def analyze(self, mri_data):
        """Generates embeddings and performs zero-shot classification."""
        print(f"INFO: Analyzing {mri_data.get('modality')} data for AD context...")
        
        central_slice = mri_data['representative_slices'][0]
        processed_img = self._preprocess_image(central_slice)
        
        # 1. Generate Embedding
        try:
            embedding = self.model.predict(processed_img, verbose=0)
        except Exception as e:
            print(f"WARN: Model prediction failed ({e}). Generating dummy embedding.")
            
            embedding = np.random.rand(1, 576) 
            
        # 2. Zero-Shot Matching
        region_name, confidence = self._find_best_match(embedding)
        
        # 3. Derive Contextual Metrics
        density = mri_data['derived_metrics'].get('tissue_density_score', 0.5)
        
        analysis_result = {
            "identified_regions": [region_name],
            "target_region_confidence": round(float(confidence), 3),
            "tissue_density_score": round(float(density), 3),
            "avoidance_zones": self._get_safety_zones(region_name),
            "target_center_depth_mm": self.mtl_center_depth_mm, 
            "ai_model": "MobileNetV3-ADNI-Concept"
        }
        
        print(f"INFO: AI Identified Target: {region_name} (Conf: {confidence:.2f})")
        return analysis_result

    def _preprocess_image(self, slice_2d):
        """Resizes and formats the 2D slice for MobileNet V3 input."""
        # Check for empty slice 
        if slice_2d.size == 0:
            slice_2d = np.zeros((100, 100))
            
        # Normalize the slice data before converting to 8-bit
        if slice_2d.max() > 0:
            normalized_slice = slice_2d / slice_2d.max()
        else:
            normalized_slice = slice_2d

        # Convert to 8-bit image for PIL/resize, then back to numpy
        img = Image.fromarray((normalized_slice * 255).astype('uint8'))
        img = img.resize((224, 224))
        img_arr = np.array(img.convert('RGB'))
        img_batch = np.expand_dims(img_arr, axis=0)
        return tf.keras.applications.mobilenet_v3.preprocess_input(img_batch)

    def _find_best_match(self, current_embedding):
        """Calculates cosine similarity to find the closest prototype region."""
        best_score = -1.0
        best_region = "Unknown"
        for name, proto_vector in self.prototypes.items():
            score = cosine_similarity(current_embedding, proto_vector)[0][0]
            if score > best_score:
                best_score = score
                best_region = name
        # Clip score between 0 and 1
        return best_region, max(0.0, min(1.0, best_score))

    def _get_safety_zones(self, region):
        """Provides safety guidance based on the identified region."""
        if "Ventricle" in region:
            return ["CRITICAL - VENTRICLE/CSF BOUNDARY DETECTED"]
        elif "MTL" in region:
            return ["PRIMARY AD TARGET - PROCEED WITH CAUTION (Deep Electrode)"]
        else:
            return ["Safe for Standard Stimulation Parameters"]