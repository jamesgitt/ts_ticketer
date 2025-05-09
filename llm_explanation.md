
## 1. Model and Tokenizer Loading

```python
hf_model = "kmcs-casulit/ts_ticket_v1.0.0.5"
tokenizer = AutoTokenizer.from_pretrained(hf_model, token=...)
model = AutoModelForCausalLM.from_pretrained(hf_model, device_map="auto", token=...).to("cuda")
```

- Loads the tokenizer and causal language model from Hugging Face Hub
- Uses `device_map="auto"` and `.to("cuda")` to automatically assign model to GPU
- Requires access token for private models

---

## 2. Prompt Template Construction

```python
ticket_prompt = """ <template with instruction and examples> """
```

- A multiline prompt that:
  - Describes the task
  - Provides two worked examples
  - Injects a new `<Ticket_Information>` section per row
- The prompt ends expecting only a valid JSON structure (no explanations)

---

## 3. Metric Calculation Function

```python
def calculate_metric_for_keys(metric_func, ground_truth, prediction):
```

- Accepts a metric function (`f1_score`, `precision_score`, or `recall_score`)
- Extracts each key (e.g., `department`, `priority`) and computes the metric independently
- Returns a dictionary mapping keys to scores

---

## 4. Main Processing Function

```python
def process_row(row):
```

This is the core logic applied to each test row:

### a. Prompt Formatting

```python
ticket_prompt.format(row["ticket_information"], "")
```

- Fills in the current row’s ticket JSON into the prompt
- Leaves a blank for model to complete the output

### b. Tokenization and Inference

```python
inputs = tokenizer([...], return_tensors="pt").to("cuda")
model_response = model.generate(**inputs, max_new_tokens=2048)
```

- Tokenizes the prompt
- Sends it to the model for generation
- Sets `max_new_tokens=2048` to bound the response size

### c. Output Decoding

```python
generated_text = tokenizer.decode(model_response[0], skip_special_tokens=True)
```

- Decodes the model’s token output into a string

### d. Response Parsing

```python
start_index = generated_text.find("<Output_Properties>") + len("<Output_Properties>")
end_index = generated_text.find("</Output_Properties>")
generated_response = generated_text[start_index:end_index].strip()
```

- Extracts text between `<Output_Properties>` tags
- Expects a valid JSON object

```python
try:
    generated_json = json.loads(generated_response)
except json.JSONDecodeError:
    generated_json = {}
```

- Attempts to parse the JSON
- Fails gracefully with empty dictionary

### e. Metric Computation

```python
ground_truth = [json.loads(row["ticket_properties"])]
prediction = [generated_json]
```

- Wraps predictions and ground truth in lists for metric functions

```python
metrics = {
    "f1_scores": calculate_metric_for_keys(f1_score, ground_truth, prediction),
    "precision_scores": calculate_metric_for_keys(precision_score, ground_truth, prediction),
    "recall_scores": calculate_metric_for_keys(recall_score, ground_truth, prediction),
    "accuracy": 1 if ground_truth == prediction else 0
}
```

- Computes metrics per key
- Accuracy is only 1 if exact JSON matches

### f. Output Storage

```python
row["ticket_properties_OUTPUT"] = json.dumps(generated_json)
row["metrics"] = json.dumps(metrics)
```

- Updates the row with model output and evaluation results

---

## 5. Batch Processing and Output

```python
test_df = pd.read_csv("ts_ticketing_test_results_v1.0.0.5.csv")
test_df = test_df.apply(process_row, axis=1)
test_df.to_csv("ts_ticketing_test_results_v1.0.0.5.csv", index=False)
```

- Loads the CSV
- Applies the `process_row()` function to each row
- Overwrites the CSV with predictions and metrics

---

## 6. Aggregated Metric Reporting (Top 100 Rows)

- Extracts accuracy and metric dictionaries from the new column `metrics`
- Calculates mean values of:
  - Accuracy
  - F1 score
  - Precision
  - Recall
- Prints all values grouped by key: `department`, `techgroup`, `category`, `subcategory`, `priority`

```python
accuracy_percentage = test_df['accuracy_value'][:100].mean() * 100
# f1_department, recall_priority, etc.
```

- Useful for quick evaluation of model quality without external tools

---

## Notes

- Assumes `ticket_information` and `ticket_properties` columns contain valid JSON strings
- Model output must contain all required keys; missing or incorrect keys will reduce accuracy
- The script is not batched — each row is processed individually, which can be slow for large datasets

---