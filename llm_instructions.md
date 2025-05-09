
---

## A. HARDWARE REQUIREMENTS
- Operating System: Ubuntu 20.04+ (or any Linux distro with GPU support)
- GPU: NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- CUDA Toolkit: Ensure it's installed (e.g., CUDA 11.8)
- RAM: Minimum 16GB system memory (32GB+ recommended)
- Storage: At least 10GB of free disk space

---

## B. FILES NEEDED
- `test_ts_ticketing.py` – The inference and evaluation script (from Colab)
- `ts_ticketing_test_results_v1.0.0.5.csv` – The test dataset CSV  
  - Must contain columns:
    - `ticket_information`: JSON string (subject, description, email)
    - `ticket_properties`: Ground truth JSON string (tags)
    
---

## C. PYTHON ENVIRONMENT SETUP
- Python Version: 3.8 or higher
- Virtual Environment (optional but recommended):
  ```bash
  python3 -m venv llm-env
  source llm-env/bin/activate
  ```
- Install Required Packages:
  ```bash
  python.exe -m pip install --upgrade pip

  pip install torch transformers pandas scikit-learn
      OR
- requirements.txt
    torch
    transformers
    pandas
    scikit-learn
  pip install -r requirements.txt
  ```
  - If using GPU, make sure to install CUDA-enabled PyTorch:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```

---

## D. HUGGING FACE MODEL ACCESS
- Model Name: `kmcs-casulit/ts_ticket_v1.0.0.5`
- Authentication Token: Hugging Face API token (must have access permission)
  - Set via environment variable:   
    ```bash
    export HUGGINGFACE_TOKEN='auth-token'
    ```
---

## E. SCRIPT FUNCTIONALITY BREAKDOWN

The script performs the following:

- **Load Model and Tokenizer**
  - From Hugging Face Hub using `AutoModelForCausalLM` and `AutoTokenizer`
  - Loads model on GPU (`device_map="auto"`)

- **Read Input CSV**
  - Loads the test dataset into a `pandas.DataFrame`

- **Prompt Formatting**
  - Constructs a structured prompt using the `ticket_information` JSON per row
  - Uses prompt examples to condition the LLM

- **Inference**
  - Calls `.generate()` to produce model output
  - Uses `max_new_tokens=2048` to limit response length
  - Decodes model output using `tokenizer.decode(...)`

- **Output Parsing**
  - Extracts JSON object from the model response
  - Handles `JSONDecodeError` fallback

- **Evaluation**
  - Compares generated output to ground truth in `ticket_properties`
  - Computes:
    - Accuracy
    - F1 score (per field)
    - Precision (per field)
    - Recall (per field)
  - Results are stored in new DataFrame columns:
    - `ticket_properties_OUTPUT`
    - `metrics` (JSON-encoded)
    - `accuracy_value`, `f1_values`, etc.

- **Export Results**
  - Saves updated DataFrame back to `ts_ticketing_test_results_v1.0.0.5.csv`
  - Prints aggregated statistics for the first 100 test cases

---

## F. OUTPUT/RESULTS PRODUCED
- Updated CSV with:
  - Generated ticket properties
  - Accuracy, F1, precision, recall
- Console output:
  - Mean accuracy and metric breakdowns (department, techgroup, category, subcategory, priority)
