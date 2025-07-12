# DeepAdvancer

**DeepAdvancer** 是一个用于批次校正、特征解释和重构表达谱的深度学习工具包，尤其适用于复杂生物分组与共享结构建模任务。其设计灵感源自空间转录组和多疾病背景下的差异分析与结构复原场景。

---

## 🚀 功能特色

- ⚙️ 基于自编码器（AutoEncoder）的表达谱重构
- 🧩 利用先验 fold change 信息学习解释性解码矩阵（sigmatrix）
- 🧠 多任务学习包括批次判别、类别判别、特征解耦
- 🔁 支持高维数据的结构保留与 batch-effect 去除
- 📊 适配多种生物学任务：DE 分析、表达结构生成、特征表示对齐

---

## 🧱 安装方式

建议使用 conda 或 virtualenv 创建独立环境：

```bash
pip install deepadvancer
```

---

## 🛠️ 使用示例

### 1. 载入表达矩阵与表型

```python
import deepadvancer

expr, pheno = load_and_process_expression_data(
    data_dir="data/",
    output_dir="output/"
)
```

### 2. 运行 FC 分析与 sigmatrix 构建

```python

train_x, class_all, batch_labels, proportions_per_feature, expected_sigmatrix = deepadvancer.run_logfc_analysis_and_generate_fc_array(
    pheno,
    expr,
    output_path="output/"
)
```

### 3. 训练自编码器模型

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

### 4. 计算 logFC 差异

```python
from deepadvancer import compute_logfc_vs_others

logfc_df = compute_logfc_vs_others(
    expression_matrix=x_recon_expr,
    phenotype_metadata=pheno,
    class_column="disease",
    target_class="psoriasis"
)
```

---

## 📦 模块说明

| 模块 | 说明 |
|------|------|
| `AutoEncoder` | 主体模型，包含 encode/decode，batch/class 判别器 |
| `sigmatrix` | 从解码层提取用于表达复原的结构性矩阵 |
| `logfc_analysis` | 融合 R 分析与 python 结构重建 |
| `training_stage1` | 三阶段损失：重构、sigmatrix、intermediate、center |
| `run_logfc_analysis_and_generate_fc_array` | 构建先验 fold change 张量与 sigmatrix 初始化 |

---

## 📄 许可证

MIT License

---

## ✉️ 作者

- **Mintian Cui**
- 联系方式: [1308318910@qq.com](mailto:1308318910@qq.com)