from pyteck.eval_model import evaluate_model

evaluate_model(
    model_name="h2o2.cti",
    spec_keys_file="spec_keys.yaml",
    dataset_file="dataset_file.txt",
    data_path="data",
    model_path="models",
    results_path="results",
)
