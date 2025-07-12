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

expr, pheno = deepadvancer.load_and_process_expression_data(
    data_dir="data/",
    output_dir="output/"
)
```

### 2. Run Fold Change Analysis and Build Sigmatrix

```python

train_x, class_all, batch_labels, proportions_per_feature, expected_sigmatrix = deepadvancer.run_logfc_analysis_and_generate_fc_array(
    pheno,
    expr,
    output_path="output/"
)
```

### 3. Train the Autoencoder Model

```python

x_recon_expr, model = deepadvancer.recon_training(
    train_x=train_x,
    class_all=class_all,
    batch_labels=batch_labels,
    proportions_per_feature=proportions_per_feature,
    expected_sigmatrix=expected_sigmatrix,
    output_path="output/",
    epochs=300
)
```

### 4. Compute logFC for Target Class

```python

logfc_df = deepadvancer.compute_logfc_vs_others(
    expression_matrix=x_recon_expr,
    phenotype_metadata=pheno,
    class_column="disease",
    target_class="psoriasis"
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
