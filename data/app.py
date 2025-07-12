import streamlit as st
import time
import os
import json
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from llama_cpp import Llama
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import regex as re
from datetime import datetime
from difflib import get_close_matches

# Set page config
st.set_page_config(
    page_title="Lunartix",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling and layout adjustments

st.markdown("""
<style>
    body {
        margin: 0;
        padding: 0;
    }
    .main {
        background-color: #1a1a1a;
        padding-bottom: 80px; /* Space for fixed input */
    }
    .stApp {
        background-color: #1a1a1a;
    }
    .header-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
        padding: 8px 10px;
        background-color: #1a1a1a;
        border-radius: 5px;
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        box-sizing: border-box;
        border-bottom: 1px solid #404040;
    }
    
    .section-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .chat-container {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 10px;
        padding: 16px 12px;
        margin: 0;
        height: 70vh; /* Increased height */
        overflow-y: auto;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        color: #ccc;
        font-size: 14px;
        scroll-behavior: smooth;
    }
    .map-container {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 10px;
        padding: 16px 12px;
        margin: 0;
        height: 70vh; /* Increased height */
        overflow-y: auto;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        color: #ccc;
        font-size: 14px;
        scroll-behavior: smooth;
    }
    .chat-message {
        margin-bottom: 13px;
        padding: 8px 13px;
        border-radius: 10px;
        max-width: 80%;
        word-wrap: break-word;
    }
    .user-message {
        background-color: #4a5568;
        color: #ffffff;
        margin-left: auto;
        text-align: right;
    }
    .ai-message {
        background-color: #2d3748;
        color: #e2e8f0;
        margin-right: auto;
        border-left: 4px solid #4299e1;
    }
    .message-avatar {
        font-size: 16px;
        margin-right: 8px;
    }
    .message-content {
        font-size: 14px;
        line-height: 1.5;
    }
    .typing-indicator {
        background-color: #2d3748;
        color: #a0aec0;
        margin-right: auto;
        border-left: 4px solid #4299e1;
        font-style: italic;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .placeholder-text {
        color: #808080;
        font-size: 14px;
        text-align: center;
        margin-top: 40px;
    }
    
    /* Fixed input styling */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #1a1a1a;
        border-top: 1px solid #404040;
        padding: 15px;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
    }
    
    .stChatInput {
        background-color: #2d2d2d !important;
        border: 1px solid #404040 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    .stChatInput > div {
        background-color: #2d2d2d !important;
    }
    
    .stChatInput input {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .chat-container::-webkit-scrollbar, .map-container::-webkit-scrollbar {
        width: 6px;
    }
    .chat-container::-webkit-scrollbar-track, .map-container::-webkit-scrollbar-track {
        background: #1a1a1a;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb, .map-container::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb:hover, .map-container::-webkit-scrollbar-thumb:hover {
        background: #5a6578;
    }
    
    /* Hide default streamlit elements */
    .stDeployButton {
        display: none !important;
    }
    
    .stDecoration {
        display: none !important;
    }
    
    footer {
        display: none !important;
    }
    
    .stToolbar {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

class IntegratedGISAnalyzer:
    def __init__(self):
        # === Set directories ===
        self.BASE_DIR = os.path.dirname(__file__)
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.OUT_DIR = os.path.join(self.BASE_DIR, "output")
        os.makedirs(self.OUT_DIR, exist_ok=True)
        
        # === Model path ===
        self.MODEL_PATH = "C:/Users/naren/OneDrive/Desktop/isro/model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        
        # === Load LLaMA model ===
        self.llm = None
        self.model_loaded = False
        
        # === Dataset metadata ===
        self.DATA_METADATA = {
            "elevation": {"range": [50, 1000], "unit": "meters"},
            "slope": {"range": [0, 45], "unit": "degrees"},
            "rainfall": {"range": [600, 1600], "unit": "mm/year"},
            "lulc": ["urban", "agriculture", "forest", "water", "barren", "grassland", "wetland"],
            "water_bodies": ["rivers", "lakes", "streams", "reservoirs", "ponds"],
            "infrastructure": ["roads", "highways", "railways", "airports"]
        }
        
        # === Load crop knowledge ===
        self.CROP_KNOWLEDGE = self._load_crop_knowledge()
        
        # === Load use cases ===
        self.USECASES = self._load_usecases()
        
        # === LULC class mapping ===
        self.lulc_class_map = {
            "urban": 1,
            "agriculture": 2,
            "forest": 3,
            "water": 4,
            "barren": 5,
            "grassland": 6,
            "wetland": 7
        }
        
        # === Base raster info (loaded once) ===
        self.base_array = None
        self.base_meta = None
        self.base_shape = None
        self.base_transform = None
        self.base_crs = None
        self._load_base_raster()
    
    def load_model(self):
        """Load LLaMA model with progress indicator"""
        if not self.model_loaded:
            try:
                if os.path.exists(self.MODEL_PATH):
                    self.llm = Llama(
                        model_path=self.MODEL_PATH,
                        n_ctx=4096,
                        n_threads=8
                    )
                    self.model_loaded = True
                    return True
                else:
                    st.error(f"Model file not found at: {self.MODEL_PATH}")
                    return False
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return False
        return True
    
    def _load_crop_knowledge(self):
        """Load crop knowledge from JSON file"""
        CROP_KNOWLEDGE_PATH = os.path.join(self.BASE_DIR, "data", "crop_knowledge.json")
        if os.path.exists(CROP_KNOWLEDGE_PATH):
            with open(CROP_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}
    
    def _load_usecases(self):
        """Load use cases from text file"""
        USECASE_PATH = os.path.join(self.BASE_DIR, "data", "usecases.txt")
        
        if not os.path.exists(USECASE_PATH):
            return []
        
        with open(USECASE_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        usecases = []
        entries = re.split(r"\n\s*\d+\.\s+\*\*Query:\*\*", content)
        
        for entry in entries[1:]:  # Skip first empty entry
            try:
                lines = entry.strip().splitlines()
                if not lines:
                    continue
                    
                query = lines[0].strip()
                
                task = ""
                cot = ""
                
                for line in lines:
                    if "**Task:**" in line:
                        task = line.split("**Task:**")[1].strip()
                    elif "**CoT:**" in line:
                        cot = line.split("**CoT:**")[1].strip()
                
                if query and task and cot:
                    usecases.append({
                        "query": query,
                        "task": task,
                        "cot": cot
                    })
            except Exception as e:
                continue
        
        return usecases
    
    def _load_base_raster(self):
        """Load base raster (elevation) for reference"""
        elevation_path = os.path.join(self.DATA_DIR, "rescaled_Elevation.tif")
        if os.path.exists(elevation_path):
            with rasterio.open(elevation_path) as base:
                self.base_array = base.read(1)
                self.base_meta = base.meta.copy()
                self.base_shape = self.base_array.shape
                self.base_transform = base.transform
                self.base_crs = base.crs
        else:
            st.warning("Base raster (elevation) not found!")
    
    def analyze_query_intent(self, user_query: str) -> dict:
        """Analyze user query to determine intent and required filters"""
        query_lower = user_query.lower()
        
        analysis = {
            "primary_intent": None,
            "features_of_interest": [],
            "spatial_context": None,
            "requires_proximity": False,
            "requires_suitability": False,
            "requires_risk_analysis": False
        }
        
        # Feature detection
        if any(word in query_lower for word in ["urban", "city", "town", "built", "settlement"]):
            analysis["features_of_interest"].append("urban")
        
        if any(word in query_lower for word in ["river", "stream", "water body", "lake", "pond"]):
            analysis["features_of_interest"].append("water")
        
        if any(word in query_lower for word in ["forest", "tree", "woodland", "jungle"]):
            analysis["features_of_interest"].append("forest")
        
        if any(word in query_lower for word in ["agriculture", "farm", "crop", "field", "cultivat"]):
            analysis["features_of_interest"].append("agriculture")
        
        if any(word in query_lower for word in ["road", "highway", "transport", "access"]):
            analysis["features_of_interest"].append("infrastructure")
        
        # Intent detection
        if any(word in query_lower for word in ["where", "locate", "find", "identify", "map"]):
            analysis["primary_intent"] = "location_query"
        
        if any(word in query_lower for word in ["suitable", "best", "optimal", "ideal"]):
            analysis["primary_intent"] = "suitability_analysis"
            analysis["requires_suitability"] = True
        
        if any(word in query_lower for word in ["flood", "risk", "hazard", "danger", "vulnerable"]):
            analysis["primary_intent"] = "risk_analysis"
            analysis["requires_risk_analysis"] = True
        
        if any(word in query_lower for word in ["near", "proximity", "close", "around"]):
            analysis["requires_proximity"] = True
        
        # Spatial context
        if "pune" in query_lower:
            analysis["spatial_context"] = "pune"
        elif "western" in query_lower:
            analysis["spatial_context"] = "western"
        elif "eastern" in query_lower:
            analysis["spatial_context"] = "eastern"
        
        return analysis
    
    def extract_crop_name(self, user_query: str) -> str:
        """Extract crop name from query"""
        for crop in self.CROP_KNOWLEDGE:
            if crop.lower() in user_query.lower():
                return crop.lower()
        return None
    
    def find_matching_usecase(self, user_query: str, analysis: dict):
        """Find best matching use case based on query and analysis"""
        if not self.USECASES:
            return None
        
        # First try exact matching
        queries = [uc["query"] for uc in self.USECASES]
        matches = get_close_matches(user_query, queries, n=3, cutoff=0.4)
        
        if matches:
            for match in matches:
                for uc in self.USECASES:
                    if uc["query"] == match:
                        if analysis["primary_intent"] == "location_query" and "identify" in uc["task"].lower():
                            return uc
                        elif analysis["requires_suitability"] and "suitability" in uc["task"].lower():
                            return uc
                        elif analysis["requires_risk_analysis"] and "risk" in uc["task"].lower():
                            return uc
                        else:
                            return uc
        
        # If no direct match, try matching by features
        for uc in self.USECASES:
            uc_query_lower = uc["query"].lower()
            for feature in analysis["features_of_interest"]:
                if feature in uc_query_lower:
                    return uc
        
        return None
    
    def call_llm(self, user_query: str) -> str:
        """Enhanced LLM prompt wrapper"""
        if not self.model_loaded:
            return "Model not loaded. Please check the model path."
        
        # Analyze query intent
        analysis = self.analyze_query_intent(user_query)
        
        # Get crop information if applicable
        crop_name = self.extract_crop_name(user_query)
        crop_prompt = ""
        if crop_name and crop_name in self.CROP_KNOWLEDGE:
            crop_info = self.CROP_KNOWLEDGE[crop_name]
            crop_prompt = f"""
Crop-specific knowledge:
- Crop: {crop_name.title()}
- Ideal Elevation: {crop_info['elevation']} meters
- Ideal Rainfall: {crop_info['rainfall']} mm/year
- Ideal Slope: {crop_info['slope']} degrees
- Notes: {crop_info['notes']}
""".strip()

        # Find matching use case
        matched = self.find_matching_usecase(user_query, analysis)
        cot_prompt = ""
        if matched:
            cot_prompt = f"""
Use case matched:
- Task: {matched['task']}
- CoT: {matched['cot']}
- Query similarity: "{matched['query']}"
"""

        # Build analysis context
        analysis_context = f"""
Query Analysis:
- Primary Intent: {analysis['primary_intent']}
- Features of Interest: {', '.join(analysis['features_of_interest']) if analysis['features_of_interest'] else 'None detected'}
- Spatial Context: {analysis['spatial_context'] or 'General'}
- Requires Proximity: {analysis['requires_proximity']}
- Requires Suitability: {analysis['requires_suitability']}
- Requires Risk Analysis: {analysis['requires_risk_analysis']}
"""

        system_prompt = """
You are an expert GIS analyst specializing in geospatial query interpretation. Your task is to analyze user queries and generate appropriate filter criteria for geospatial analysis.

Key principles:
1. Only include filters that are directly relevant to the user's query
2. For simple location queries (like "where are urban areas"), only return the relevant LULC classes
3. For suitability analysis, include elevation, slope, rainfall, and LULC as needed
4. For risk analysis, focus on hazard-relevant parameters
5. Always provide clear reasoning for your choices
"""

        metadata_description = f"""
Available data layers and ranges:
- Elevation: {self.DATA_METADATA['elevation']['range'][0]}–{self.DATA_METADATA['elevation']['range'][1]} {self.DATA_METADATA['elevation']['unit']}
- Slope: {self.DATA_METADATA['slope']['range'][0]}–{self.DATA_METADATA['slope']['range'][1]} {self.DATA_METADATA['slope']['unit']}
- Rainfall: {self.DATA_METADATA['rainfall']['range'][0]}–{self.DATA_METADATA['rainfall']['range'][1]} {self.DATA_METADATA['rainfall']['unit']}
- LULC classes: {', '.join(self.DATA_METADATA['lulc'])}

""".strip()

        user_prompt = f"""
{analysis_context}

{crop_prompt}

{cot_prompt}

User query: "{user_query}"

Based on the query analysis and use case matching, generate appropriate filter criteria:

IMPORTANT RULES:
1. For simple location queries (like "where are urban areas"), ONLY return the relevant features:
   - Return only LULC classes that match the query
   - Do NOT include elevation, slope, or rainfall unless specifically relevant

2. For suitability analysis, include relevant parameters:
   - Elevation, slope, rainfall ranges as needed
   - Appropriate LULC classes
   - Proximity buffers if needed

3. For risk analysis, focus on hazard-relevant parameters:
   - Elevation and slope for flood/landslide risk
   - Rainfall for climate-related risks
   - Relevant LULC classes

4. Always provide reasoning in the notes field

Output format:
{{
  "task": "location_query|suitability_analysis|risk_analysis",
  "criteria": {{
    // Only include relevant parameters
    "lulc": ["relevant_classes"],
    "elevation": [min, max],  // Only if needed
    "slope": [min, max],      // Only if needed
    "rainfall": [min, max],   // Only if needed
    
  }},
  "notes": "Explanation of filter choices and reasoning"
}}

Remember: Only include filters that are directly relevant to answering the user's question!
"""

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": metadata_description + "\n\n" + user_prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
                stop=["</s>"]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def extract_json(self, text: str):
        """Extract JSON from LLM response with robust parsing"""
        try:
            # Method 1: Find JSON block using regex
            json_patterns = [
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Simple nested JSON
                r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}",     # Alternative pattern
                r"\{[\s\S]*?\}(?=\s*$|\s*\n\s*[A-Z])"  # JSON block before text
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        # Clean the JSON string
                        json_str = self._clean_json_string(match)
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            # Method 2: Try to extract content between first { and last }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = text[start_idx:end_idx+1]
                json_str = self._clean_json_string(json_str)
                return json.loads(json_str)
            
            # Method 3: Manual parsing fallback
            return self._manual_json_extraction(text)
            
        except Exception as e:
            return self._manual_json_extraction(text)
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string by removing comments and fixing common issues"""
        # Remove single-line comments (// comment)
        json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
        
        # Remove multi-line comments (/* comment */)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix unquoted property names
        json_str = re.sub(r'(\w+)(\s*:)', r'"\1"\2', json_str)
        
        # Fix single quotes to double quotes
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        return json_str
    
    def _manual_json_extraction(self, text: str) -> dict:
        """Fallback manual JSON extraction for when parsing fails"""
        # Try to extract key information manually
        result = {
            "task": "suitability_analysis",
            "criteria": {},
            "notes": "Extracted using manual parsing"
        }
        
        # Extract task type
        if "location_query" in text.lower():
            result["task"] = "location_query"
        elif "risk_analysis" in text.lower():
            result["task"] = "risk_analysis"
        
        # Extract LULC classes
        lulc_classes = []
        for lulc_class in self.DATA_METADATA["lulc"]:
            if lulc_class in text.lower():
                lulc_classes.append(lulc_class)
        
        if lulc_classes:
            result["criteria"]["lulc"] = lulc_classes
        
        # Extract elevation range
        elevation_match = re.search(r'"elevation":\s*\[(\d+),\s*(\d+)\]', text)
        if elevation_match:
            result["criteria"]["elevation"] = [int(elevation_match.group(1)), int(elevation_match.group(2))]
        
        # Extract slope range
        slope_match = re.search(r'"slope":\s*\[(\d+),\s*(\d+)\]', text)
        if slope_match:
            result["criteria"]["slope"] = [int(slope_match.group(1)), int(slope_match.group(2))]
        
        # Extract rainfall range
        rainfall_match = re.search(r'"rainfall":\s*\[(\d+),\s*(\d+)\]', text)
        if rainfall_match:
            result["criteria"]["rainfall"] = [int(rainfall_match.group(1)), int(rainfall_match.group(2))]
        
        # Extract notes
        notes_match = re.search(r'"notes":\s*"([^"]*)"', text)
        if notes_match:
            result["notes"] = notes_match.group(1)
        
        return result
    
    def load_and_mask_raster(self, filename, value_range):
        """Load and mask raster based on value range"""
        path = os.path.join(self.DATA_DIR, filename)
        try:
            with rasterio.open(path) as src:
                data = src.read(1)
                mask = (data >= value_range[0]) & (data <= value_range[1])
                return mask
        except Exception as e:
            st.error(f"Error loading raster {filename}: {str(e)}")
            return None
    
    def process_geospatial_analysis(self, criteria):
        """Process geospatial analysis based on criteria"""
        # Initialize mask list
        mask_list = []
        
        # Process elevation criteria
        if "elevation" in criteria:
            mask_elevation = self.load_and_mask_raster("rescaled_Elevation.tif", criteria["elevation"])
            if mask_elevation is not None:
                mask_list.append(mask_elevation)
        
        # Process slope criteria
        if "slope" in criteria:
            mask_slope = self.load_and_mask_raster("cleaned_slope.tif", criteria["slope"])
            if mask_slope is not None:
                mask_list.append(mask_slope)
        
        # Process rainfall criteria
        if "rainfall" in criteria:
            mask_rainfall = self.load_and_mask_raster("rescaled_rainfall.tif", criteria["rainfall"])
            if mask_rainfall is not None:
                mask_list.append(mask_rainfall)
        
        # Process LULC criteria
        if "lulc" in criteria:
            lulc_path = os.path.join(self.DATA_DIR, "resampled_lulc.tif")
            try:
                with rasterio.open(lulc_path) as lulc_src:
                    lulc_data = lulc_src.read(1)
                    lulc_resampled = np.empty(shape=self.base_shape, dtype=lulc_data.dtype)

                    reproject(
                        source=lulc_data,
                        destination=lulc_resampled,
                        src_transform=lulc_src.transform,
                        src_crs=lulc_src.crs,
                        dst_transform=self.base_transform,
                        dst_crs=self.base_crs,
                        resampling=Resampling.nearest
                    )

                lulc_values = [self.lulc_class_map[c] for c in criteria["lulc"]]
                mask_lulc = np.isin(lulc_resampled, lulc_values)
                mask_list.append(mask_lulc)
            except Exception as e:
                st.error(f"Error processing LULC data: {str(e)}")
        
        # Combine all masks
        if mask_list:
            final_mask = np.logical_and.reduce(mask_list)
            return final_mask
        else:
            return None
    
    def visualize_results(self, final_mask):
        """Create visualization with Pune villages"""
        try:
            village_path = os.path.join(self.DATA_DIR, "pune_villages.geojson")
            villages = None
            if os.path.exists(village_path):
                villages = gpd.read_file(village_path).to_crs(self.base_crs)
            
            # Create polygons from mask
            shapes_gen = shapes(final_mask.astype(np.uint8), transform=self.base_transform)
            polygons = [
                {"geometry": shape(geom), "properties": {"value": val}}
                for geom, val in shapes_gen if val == 1
            ]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot village boundaries if available
            if villages is not None:
                villages.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.5, alpha=0.7, label="Pune Villages")
            
            # Plot suitability zones
            if polygons:
                gdf = gpd.GeoDataFrame.from_features(polygons, crs=self.base_crs)
                gdf.plot(ax=ax, color='green', alpha=0.6, edgecolor='darkgreen', linewidth=0.3, label="Suitable Zones")
            
            ax.set_title("Geospatial Analysis Results - Pune District", fontsize=16, fontweight='bold')
            ax.set_xlabel("Longitude", fontsize=12)
            ax.set_ylabel("Latitude", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches='tight')
                plt.close()
                return tmp.name
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return None

def main():
    """Main function to run the Streamlit application"""
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = IntegratedGISAnalyzer()
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_visualization' not in st.session_state:
        st.session_state.current_visualization = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Header
    st.markdown('<div class="header-title">🌙 Lunartix - GIS Query Assistant</div>', unsafe_allow_html=True)
    
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Load model if not already loaded
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            st.session_state.model_loaded = st.session_state.analyzer.load_model()
    
    if not st.session_state.model_loaded:
        st.error("Failed to load the AI model. Please check the model path and try again.")
        return
    
    # Create two columns for chat and map
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-title">💬 Chat Interface</div>', unsafe_allow_html=True)
        
        # Display chat messages
        chat_html = '<div class="chat-container">'
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    chat_html += f'''
                    <div class="chat-message user-message">
                        <div class="message-content">
                            <span class="message-avatar">🧑‍💻</span>
                            {content}
                        </div>
                    </div>
                    '''
                else:
                    chat_html += f'''
                    <div class="chat-message ai-message">
                        <div class="message-content">
                            <span class="message-avatar">🤖</span>
                            {content}
                        </div>
                    </div>
                    '''
        else:
            chat_html += '<div class="placeholder-text">Start a conversation by typing your query below...</div>'
        
        # Only show thinking indicator for AI, not for user input
        if st.session_state.processing:
            chat_html += '''
            <div class="chat-message typing-indicator">
                <div class="message-content">
                    <span class="message-avatar">🤖</span>
                    Thinking...
                </div>
            </div>
            '''
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-title">🗺️ Analysis Visualization</div>', unsafe_allow_html=True)
        
        if st.session_state.current_visualization:
            st.image(st.session_state.current_visualization, caption="Geospatial Analysis Result", use_column_width=True)
        else:
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            st.write("No analysis visualization available yet.\n\nAsk a query to generate results.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fixed input at bottom - outside the columns
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # Chat input
    chat_input = st.chat_input("Ask me anything about GIS, locations, or geospatial analysis...")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process input
    if chat_input and not st.session_state.processing:
        st.session_state.chat_history.append({"role": "user", "content": chat_input})
        st.session_state.processing = True
        st.rerun()
    
    # Process query if in processing state
    if st.session_state.processing:
        last_user_msg = None
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if last_user_msg:
            response = st.session_state.analyzer.call_llm(last_user_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Try to extract JSON criteria for geospatial analysis
            criteria = st.session_state.analyzer.extract_json(response)
            
            final_mask = None
            if criteria and isinstance(criteria, dict) and "criteria" in criteria:
                final_mask = st.session_state.analyzer.process_geospatial_analysis(criteria["criteria"])
            
            if final_mask is not None:
                vis_path = st.session_state.analyzer.visualize_results(final_mask)
                if vis_path:
                    st.session_state.current_visualization = vis_path
                else:
                    st.session_state.current_visualization = None
            else:
                st.session_state.current_visualization = None
            
            st.session_state.processing = False
            st.rerun()

if __name__ == "__main__":
    main()