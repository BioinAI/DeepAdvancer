# DeepAdvancer

**DeepAdvancer**  is a deep learning toolkit for batch correction and expression reconstruction, specifically designed for biologically complex class structures.

---

## 🚀 Features

- ⚙️ Autoencoder-based transcriptome reconstruction
- 🧩 Learns interpretable decoding matrix (sigmatrix) using prior fold-change information
- 🧠 Multi-task learning including batch classification, class prediction, and feature disentanglement
- 🔁 Preserves biological structure while removing batch effects from high-dimensional data
- 📊 Supports various downstream biological tasks such as differential expression analysis, feature alignment, and expression synthesis

---

## 🧱 Installation

It is recommended to use conda or virtualenv to create an isolated environment:

```bash
pip install deepadvancer
```

---

## 🛠️ Quick Usage

### 1. Load Expression Matrix and Phenotype Metadata

```python
import deepadvancer

expr_matrix, pheno_data = load_and_process_expression_data(
    data_dir=data_path,
    output_dir=data_path ,
    gene_threshold_ratio=0.8,
    dataset_gene_coverage_threshold=0.8
)

```

### 2. Run Fold Change Analysis and Build Sigmatrix

💡 Note: Before running this step, make sure your saved phenotype metadata includes a 'disease' column indicating class labels (e.g., psoriasis, lupus, etc.). If you've already saved the expression and metadata files with correct formatting, you can skip Step 1 and directly load them using pd.read_csv() or similar.

```python

train_x, class_all, batch_labels, proportions_per_feature, expected_sigmatrix = run_logfc_analysis_and_generate_fc_array(
    sample_metadata=sample_metadata,
    expr_matrix=expr_matrix,
    output_path=data_path
)
```

### 3. Train the Autoencoder Model

```python
x_recon_expression_matrix, model = recon_training(
    expr_matrix=expr_matrix,
    train_x=train_x,
    class_all=class_all,
    batch_labels=batch_labels,
    proportions_per_feature=proportions_per_feature,
    expected_sigmatrix=expected_sigmatrix,
    output_path=data_path
    batch_size=128,
    epochs=300
)
```

### 4. Compute logFC for Target Class

```python

logfc_df = deepadvancer.compute_logfc_vs_others(
    expression_matrix=x_recon_expression_matrix,
    phenotype_metadata=sample_metadat_cleaned,
    class_column="disease",
    target_class="psoriasis"
)
```


### 5. Compute logFC Between Two Classes

```python

logfc_df = deepadvancer.compute_logfc_between_classes(
    expression_matrix=x_recon_expression_matrix,
    phenotype_metadata=sample_metadat_cleaned,
    class_column='disease',
    class1='healthy',
    class2='psoriasis'
)

```

### 6. Projection of Matrix

```python
deepadvancer.plot_pca_by_class(
    expression_df=x_recon_expression_matrix,,
    class_series=sample_metadat_cleaned['disease'],
    jitter_scale=0.8,
    point_size=15,
    max_legend_classes=50,
    title="Expression Matrix",
    figsize=(12, 10),
    alpha=0.8,
    save_path=None 
)
```

---

## 📦 Module Overview

|Module | Description |
|------|------|
| `load_and_process_expression_data` | Integrates raw expression matrix and phenotype metadata into a unified format |
| `run_logfc_analysis_and_generate_fc_array` | Prepares fold-change tensor and interpretable sigmatrix via shared logFC analysis |
| `recon_training` | Trains the adversarial autoencoder with batch correction and structure alignment |
| `compute_logfc_between_classes` | Calculates log2 fold change between any two specified classes |
| `compute_logfc_vs_others` | Computes log2 fold change of one class against all other classes |

---

## 📄 License

MIT License

---

## ✉️ Author

- **Mintian Cui**
- Contact: [1308318910@qq.com](mailto:1308318910@qq.com)
