import os
import pandas as pd
import numpy as np
from collections import Counter
import glob
from sklearn.preprocessing import LabelEncoder
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as robjects
from collections import defaultdict
from scipy.stats import pearsonr
import torch.nn as nn
import torch
import random
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from importlib import resources

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_process_expression_data(
    data_dir,
    output_dir=None,
    gene_threshold_ratio=0.8,
    dataset_gene_coverage_threshold=0.8
):
    """
    读取并整合一个目录下的多个表达矩阵和表型文件，并完成清洗和归一化。

    参数：
        data_dir: str
            目录路径，每个子文件夹包含一个表达矩阵（包含"expression"）和一个表型文件（包含"phenotype"），均为.csv格式。
        output_dir: str or None
            如果设置，将结果保存到该目录下。
        gene_threshold_ratio: float
            定义“共有基因”的阈值（默认80%数据集中都存在的基因）。
        dataset_gene_coverage_threshold: float
            每个数据集包含多少比例共有基因才保留（默认80%）。

    返回：
        final_expression_matrix_scaled: pd.DataFrame
            归一化后的表达矩阵
        phenotype_metadat_scaled: pd.DataFrame
            对应的样本表型数据
    """

    expression_data = {}
    phenotype_data = {}

    # 读取所有表达和表型文件
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            expr_files = glob(os.path.join(folder_path, '*expression*.csv'))
            pheno_files = glob(os.path.join(folder_path, '*phenotype*.csv'))

            if expr_files and pheno_files:
                expr_file = expr_files[0]
                pheno_file = pheno_files[0]

                expr_df = pd.read_csv(expr_file, index_col=0)
                pheno_df = pd.read_csv(pheno_file, index_col=0)

                expression_data[folder_name] = expr_df
                phenotype_data[folder_name] = pheno_df

    print(f"共读取到 {len(expression_data)} 个数据集。")

    # 找出共有基因
    gene_sets = [set(df.index) for df in expression_data.values()]
    total_datasets = len(gene_sets)
    gene_counter = Counter(g for genes in gene_sets for g in genes)
    gene_threshold = int(total_datasets * gene_threshold_ratio)
    common_genes = [gene for gene, count in gene_counter.items() if count >= gene_threshold]
    print(f"共有基因数（出现在 ≥{gene_threshold_ratio:.0%} 数据集中）: {len(common_genes)}")

    # 筛选保留的数据集
    filtered_expression_data = {}
    filtered_phenotype_data = {}

    for dataset_name, expr_df in expression_data.items():
        gene_overlap = set(common_genes).intersection(expr_df.index)
        overlap_ratio = len(gene_overlap) / len(common_genes)

        if overlap_ratio >= dataset_gene_coverage_threshold:
            filtered_expr_df = expr_df.loc[list(gene_overlap)].copy()
            filtered_expression_data[dataset_name] = filtered_expr_df
            filtered_phenotype_data[dataset_name] = phenotype_data[dataset_name]
            print(f"✅ 保留数据集: {dataset_name}（共有基因比例 {overlap_ratio:.2%}）")
        else:
            print(f"❌ 丢弃数据集: {dataset_name}（共有基因比例 {overlap_ratio:.2%}）")

    # 合并表达矩阵
    #expression_combined = pd.concat(
    #    [df.reindex(common_genes).fillna(0) for df in filtered_expression_data.values()],
     #   axis=1
    #)
    
    # 先将所有表达矩阵对齐至 common_genes（保留 NaN）
    aligned_dfs = [df.reindex(common_genes) for df in filtered_expression_data.values()]

    # 拼接为临时大矩阵（列是样本，行是基因）
    expression_combined = pd.concat(aligned_dfs, axis=1)

    # 按行（每个基因）填补 NaN 为该基因的平均表达（跨样本）
    expression_combined = expression_combined.apply(lambda row: row.fillna(row.mean()), axis=1)
   
    

    # 合并表型信息
    all_pheno = []
    for dataset_name, df in filtered_phenotype_data.items():
        df = df.copy()
        df['source_dataset'] = dataset_name
        all_pheno.append(df)

    phenotype_combined = pd.concat(all_pheno, axis=0)

    # 去重
    expression_combined = expression_combined.loc[:, ~expression_combined.columns.duplicated()]
    phenotype_combined = phenotype_combined.loc[~phenotype_combined.index.duplicated(keep='first')]

    # 对齐样本
    shared_samples = expression_combined.columns.intersection(phenotype_combined.index)
    expression_combined = expression_combined[shared_samples]
    phenotype_combined = phenotype_combined.loc[shared_samples]

    # 清洗变量名
    sample_metadat_cleaned = phenotype_combined
    expression_matrix_cleaned = expression_combined

    # 预处理表达数据
    grouped = sample_metadat_cleaned.groupby('source_dataset')
    processed_datasets = []

    for i, (source, group) in enumerate(grouped, 1):
        print(f"Processing dataset {i}: {source}")
        sample_ids = group.index
        sub_matrix = expression_matrix_cleaned[sample_ids].copy()

        #same_value_genes = sub_matrix.nunique(axis=1) == 1
        #sub_matrix.loc[same_value_genes] = 0

        def handle_outliers(matrix):
            mean_value = np.nanmean(matrix.values)
            max_value = np.nanmax(matrix.values)
            if max_value > 10 * mean_value:
                lower_bound = np.nanquantile(matrix.values, 0.01)
                upper_bound = np.nanquantile(matrix.values, 0.99)
                print(f"Clipping values outside [{lower_bound}, {upper_bound}].")
                return matrix.clip(lower=lower_bound, upper=upper_bound)
            else:
                return matrix

        sub_matrix = handle_outliers(sub_matrix)

        def needs_log_transform(matrix):
            qx = np.nanquantile(matrix.values.flatten(), [0.0, 0.25, 0.5, 0.75, 0.99, 1.0])
            return (qx[5] > 100) or ((qx[5] - qx[0] > 50) and (qx[1] > 0))

        if needs_log_transform(sub_matrix):
            print(f"Dataset {i}: {source} requires log transformation.")
            #sub_matrix[sub_matrix <= 0] = np.nan
            #sub_matrix = np.log2(sub_matrix)
            #sub_matrix = sub_matrix.fillna(0)
            min_val = sub_matrix.min().min()
            if min_val <= 0:
                shift = abs(min_val) + 1e-3
                sub_matrix += shift
            sub_matrix = np.log2(sub_matrix)


        processed_datasets.append(sub_matrix)

    # 合并处理后的表达数据
    final_expression_matrix = pd.concat(processed_datasets, axis=1)
    print("All datasets processed and combined.")

    # 删除表达值相同的基因（80% 样本相同值）
    rows_to_remove = []
    for row in final_expression_matrix.index:
        value_counts = final_expression_matrix.loc[row].value_counts(normalize=True)
        if value_counts.max() >= 0.6:
            rows_to_remove.append(row)

    final_expression_matrix = final_expression_matrix.drop(index=rows_to_remove)
    #sample_metadat_cleaned = sample_metadat_cleaned.drop(index=rows_to_remove)
    print(f"Removed {len(rows_to_remove)} rows with low expression variance.")

    # 按数据集进行 min-max 归一化
    normalized_datasets = []

    for dataset_name, group in sample_metadat_cleaned.groupby('source_dataset'):
        sample_ids = group.index.tolist()
        valid_samples = [s for s in sample_ids if s in final_expression_matrix.columns]
        if not valid_samples:
            continue
        sub_expr = final_expression_matrix[valid_samples].copy()
        dataset_max = sub_expr.values.max()
        dataset_min = sub_expr.values.min()
        if dataset_max == dataset_min:
            print(f"⚠️ 丢弃无信息数据集: {dataset_name}")
            continue
        sub_expr_scaled = sub_expr#(sub_expr - dataset_min) / (dataset_max - dataset_min)
        normalized_datasets.append(sub_expr_scaled)

    final_expression_matrix_scaled = pd.concat(normalized_datasets, axis=1)

    # 保留样本并对齐
    retained_samples = final_expression_matrix_scaled.columns
    phenotype_metadat_scaled = sample_metadat_cleaned.loc[
        sample_metadat_cleaned.index.intersection(retained_samples)
    ]
    phenotype_metadat_scaled = phenotype_metadat_scaled.loc[retained_samples]

    print(f"归一化后表达矩阵维度: {final_expression_matrix_scaled.shape}")
    print(f"归一化后表型数据维度: {phenotype_metadat_scaled.shape}")
    
     # Step 9: 保存到输出路径
    final_expression_matrix_scaled.to_csv(os.path.join(output_dir, "expression_matrix_scaled.csv"))
    phenotype_metadat_scaled.to_csv(os.path.join(output_dir, "phenotype_metadat_scaled.csv"))
    print(f"✅ 保存成功: 文件已写入 {output_dir}")

    return final_expression_matrix_scaled, phenotype_metadat_scaled

def showloss(loss_list, title="Loss Curve"):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_list)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def calculate_gene_expression(adjusted_z, sigmatrix, class_labels, calculate_intermediate=True):
    """
    根据调整后的 Z 和 sigmatrix 计算基因表达值。
    :param adjusted_z: 调整后的特征，形状为 (batch_size, feature_dim, num_genes)
    :param sigmatrix: 矩阵，形状为 (feature_dim, num_genes)
    :param class_labels: 样本的类别标签，形状为 (batch_size,)
    :param calculate_intermediate: 是否计算中间结果，默认为 True
    :return: 基因表达值，形状为 (batch_size, num_genes)
             如果 calculate_intermediate 为 True，还返回 intermediate_result
    """
    batch_size, feature_dim, num_genes = adjusted_z.size()

    # 初始化基因表达张量
    gene_expression = torch.zeros((batch_size, num_genes), device=adjusted_z.device)

    # 如果需要计算 intermediate_result，则初始化它
    if calculate_intermediate:
        intermediate_result = torch.zeros((batch_size, feature_dim, num_genes), device=adjusted_z.device)

    for i in range(batch_size):
        # 当前样本的调整特征 (feature_dim, num_genes)
        current_adjusted_z = adjusted_z[i]  # (feature_dim, num_genes)
        sample_class = class_labels[i].item()  # 当前样本的类别标签
        #print(sample_class)
        
        #if calculate_intermediate:
        #    feature_weights = torch.ones(feature_dim, device=adjusted_z.device)  # 初始化特征权重为1

            # 按特征逐行计算
       #     for j in range(feature_dim):
                # 如果样本类别属于当前特征的类标签集合，将比例调整为8倍
       #         if sample_class in feature_class_sets[j]:
      #              feature_weights[j] = 100.0

                # 当前特征值和 sigmatrix 对应行相乘
      #          intermediate_result[i, j] = current_adjusted_z[j] * sigmatrix[j]  # (num_genes,)

            # 对特征权重进行归一化
       #     feature_weights /= feature_weights.sum()
            #print(feature_weights)
            
            # 对所有特征进行累加，按照归一化的权重调整
       #     gene_expression[i] = (intermediate_result[i] * feature_weights.unsqueeze(-1)).sum(dim=0)
      #  else:
            # 如果不需要中间结果，直接计算基因表达值
     #       gene_expression[i] = torch.matmul(current_adjusted_z, sigmatrix.T).sum(dim=0)

        if calculate_intermediate:
            # 按特征逐行计算
            for j in range(feature_dim):
                # 当前特征值和 sigmatrix 对应行相乘
                intermediate_result[i, j] = current_adjusted_z[j] * sigmatrix[j]  # (num_genes,)

            # 对所有特征进行累加，调整特征1的比例为 3 倍
            gene_expression[i] = intermediate_result[i].sum(dim=0) + 3 * intermediate_result[i, 0] # + 0.1 * b4 # 特征1比例变为 3 倍
        else:
            # 如果不需要中间结果，直接计算基因表达值
            gene_expression[i] = torch.matmul(current_adjusted_z, sigmatrix.T).sum(dim=0) #+ b4 
            

    if calculate_intermediate:
        return gene_expression, intermediate_result
    else:
        return gene_expression 
    
def adjust_features(z, class_sub, proportions_per_feature):
    """
    根据 class_sub 和 proportions_per_feature 为每个特征构建调整矩阵并应用到 z。
    :param z: 编码后的特征，形状为 (batch_size, feature_dim)
    :param class_sub: 当前批次的类标签，形状为 (batch_size,)
    :param proportions_per_feature: 每个特征的调整比例，是一个 (feature_dim, num_classes, num_genes) 的张量
    :return: 调整后的特征，返回一个 (batch_size, feature_dim, num_genes) 的张量
    """
    # 确保 proportions_per_feature 与 z 在同一设备上
    proportions_per_feature = proportions_per_feature.to(z.device)
    
    # 获取张量形状
    batch_size, feature_dim = z.size()  # (batch_size, feature_dim)
    num_classes, num_genes = proportions_per_feature.size(1), proportions_per_feature.size(2)

    # 创建 one-hot 编码矩阵，用于选择当前类标签的调整比例
    class_one_hot = F.one_hot(class_sub, num_classes=num_classes).float()  # (batch_size, num_classes)
    
    # 选择每个样本对应类标签的调整比例
    # 使用张量乘法：class_one_hot -> (batch_size, num_classes)
    # proportions_per_feature -> (feature_dim, num_classes, num_genes)
    # 输出的 selected_proportions -> (batch_size, feature_dim, num_genes)
    selected_proportions = torch.einsum('bc,fcg->bfg', class_one_hot, proportions_per_feature)

    # 将编码特征 z 扩展为与 selected_proportions 相同的形状
    z_expanded = z.unsqueeze(-1)  # (batch_size, feature_dim, 1)
    
    # 广播计算调整后的特征值
    adjusted_features = z_expanded * selected_proportions  # (batch_size, feature_dim, num_genes)
        # 添加非线性激活函数
    #adjusted_features = F.relu(adjusted_features)  # ReLU 激活函数

    return adjusted_features 

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.CELU(),
            nn.Linear(64, 32),
            nn.CELU(),
            nn.Linear(32, num_outputs),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_batches,num_classes):
        super().__init__()
        self.name = 'ae'
        self.state = 'train' # or 'test'
        self.inputdim = input_dim
        self.outputdim = output_dim
        self.num_batches = num_batches
        self.num_classes = num_classes
        self.encoder = nn.Sequential(nn.Dropout(),
                                     nn.Linear(self.inputdim, 512),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(512, 256),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(256, 128),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),
                                     
                                     nn.Linear(64, output_dim),
                                     )
        

        self.decoder = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     #nn.CELU(),
                                     nn.Linear(64, 128, bias=False),
                                     #nn.CELU(),
                                     nn.Linear(128, 256, bias=False),
                                     #nn.CELU(),
                                     nn.Linear(256, 512, bias=False),
                                     #nn.CELU(),
                                     nn.Linear(512, self.inputdim, bias=False)
                                     #nn.Sigmoid()  # Added Sigmoid activation here
                                    )

        
        # Batch discriminator
        self.batch_discriminator = Discriminator(output_dim, num_batches)

        # Class discriminator
        self.class_discriminator = Discriminator(output_dim, num_classes)  
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self,x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x/x_sum

    #def sigmatrix(self):
    #    w0 = (self.decoder[0].weight.T)
    #    w1 = (self.decoder[1].weight.T)
    #    w2 = (self.decoder[2].weight.T)
    #    w3 = (self.decoder[3].weight.T)
    #    w4 = (self.decoder[4].weight.T)
    #    w01 = (torch.mm(w0, w1))
     #   w02 = (torch.mm(w01, w2))
    #    w03 = (torch.mm(w02, w3))
    #    w04 = (torch.mm(w03, w4))
     #   return F.relu(w04)
    
    def sigmatrix_with_bias(self):
        # 获取每一层的权重和偏置
        w0 = self.decoder[0].weight.T 
        #b0 = self.decoder[0].bias    
        w1 = self.decoder[1].weight.T  
        #b1 = self.decoder[1].bias     
        w2 = self.decoder[2].weight.T 
        #b2 = self.decoder[2].bias     
        w3 = self.decoder[3].weight.T  
        #b3 = self.decoder[3].bias     
        w4 = self.decoder[4].weight.T  
        #b4 = self.decoder[4].bias    

        # 逐层计算并应用偏置
        w01 = torch.mm(w0, w1)  # 形状: (z_dim, 128)
        #out01 = w01 + b1.unsqueeze(0)  # 偏置加到第一层输出

        w02 = torch.mm(w01, w2)  # 形状: (z_dim, 256)
        #out02 = w02 + b2.unsqueeze(0)  # 偏置加到第二层输出

        w03 = torch.mm(w02, w3)  # 形状: (z_dim, 512)
        #out03 = w03 + b3.unsqueeze(0)  # 偏置加到第三层输出

        w04 = torch.mm(w03, w4)  # 形状: (z_dim, inputdim)
        #out04 = w04 + b4.unsqueeze(0)  # 偏置加到第四层输出

        return w04  # 返回最终的结果


   # def forward(self, x, class_sub, proportions_per_feature):
    #    sigmatrix = self.sigmatrix()
    #    z = self.encode(x)
        #print(z.size())       
        
        # Predict batch and class
    #    batch_pred = self.batch_discriminator(z)
    #    class_pred = self.class_discriminator(z)

        # 调用函数
    #    adjusted_features = adjust_features(z, class_sub, proportions_per_feature)        
            
    #    x_recon, intermediate_result = calculate_gene_expression(adjusted_features, sigmatrix)

    #    return x_recon, z, batch_pred, class_pred, sigmatrix, intermediate_result
    
    def forward(self, x, expected_sigmatrix, class_sub, proportions_per_feature, calculate_intermediate):
        sigmatrix = self.sigmatrix_with_bias()
        #print(b04.shape)
        z = self.encode(x)
        #x_recon1 = self.decode(z)

        # Predict batch and class
        batch_pred = self.batch_discriminator(z)
        class_pred = self.class_discriminator(z)

        # 调用函数
        calculate_intermediate = self.state == 'train'
        adjusted_features = adjust_features(z, class_sub, proportions_per_feature)

        #x_recon = calculate_gene_expression(adjusted_features, sigmatrix, calculate_intermediate)
        x_recon = calculate_gene_expression(adjusted_features, sigmatrix ,class_sub, calculate_intermediate)

        if calculate_intermediate:
            gene_expression, intermediate_result = x_recon
            return gene_expression, z, batch_pred, class_pred, sigmatrix, intermediate_result
        else:
            gene_expression = x_recon1
            return gene_expression, z, batch_pred, class_pred, sigmatrix    

class SimDataset1(Dataset):
    def __init__(self, X, class_labels_all, batch_labels_all):
        self.X = X  # 输入数据
        self.class_labels_all = class_labels_all  # 类别标签
        self.batch_labels_all = batch_labels_all  # 批次标签

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        class_label_all = torch.tensor(self.class_labels_all[index]).long()
        batch_label_all = torch.tensor(self.batch_labels_all[index]).long()

        # 将数据移到正确的设备上
        return x.to(device), class_label_all.to(device), batch_label_all.to(device)
    
def training_stage1(model, train_loader, optimizer_ae, optimizer_disc_batch, optimizer_disc_class, 
                    proportions_per_feature, expected_sigmatrix, epochs=128):
    """
    训练阶段 1
    :param model: AutoEncoder 模型
    :param train_loader: 数据加载器
    :param optimizer_ae: 自编码器优化器
    :param optimizer_disc_batch: 批次判别器优化器
    :param optimizer_disc_class: 类别判别器优化器
    :param proportions_per_feature: 每个特征的调整比例，形状为 (feature_dim, num_classes, num_genes)
    :param expected_sigmatrix: 外部传入的 sigmatrix，形状为 (feature_dim, num_genes)
    :param epochs: 训练轮数
    :return: 训练后的模型及各损失列表
    """
    model.train()
    model.state = 'train'

    # 初始化损失函数
    criterion_recon = nn.MSELoss()
    criterion_disc = nn.CrossEntropyLoss()
    criterion_sigmatrix = nn.MSELoss()
    criterion_intermediate = nn.MSELoss()

    # 初始化损失记录
    recon_loss_all = []
    adv_loss = []
    class_loss = []
    sigmatrix_loss_all = []
    intermediate_loss_all = []
  
    # 在训练循环外部初始化类中心和样本计数器
    class_centers = {}  # 用于存储每个类的中心，键为类别索引，值为中心向量
    class_sample_counts = {}  # 用于存储每个类的样本数量

    for epoch in tqdm(range(epochs)):
        for x, class_label, batch_label in train_loader:
            
            ### Step 1: 训练批次判别器 ###
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = False
            for param in model.batch_discriminator.parameters():
                param.requires_grad = True
            for param in model.class_discriminator.parameters():
                param.requires_grad = False

            optimizer_disc_batch.zero_grad()

            z = model.encode(x)
            batch_pred = model.batch_discriminator(z)
            batch_disc_loss = criterion_disc(batch_pred, batch_label)
            batch_disc_loss.backward()
            optimizer_disc_batch.step()

            adv_loss.append(batch_disc_loss.item())

            ### Step 2: 训练类别判别器 ###
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = False
            for param in model.batch_discriminator.parameters():
                param.requires_grad = False
            for param in model.class_discriminator.parameters():
                param.requires_grad = True

            optimizer_disc_class.zero_grad()

            z = model.encode(x)
            class_pred = model.class_discriminator(z)
            class_disc_loss = criterion_disc(class_pred, class_label)
            class_disc_loss.backward()
            optimizer_disc_class.step()

            class_loss.append(class_disc_loss.item())

            ### Step 3: 训练自编码器 ###
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = True
            for param in model.batch_discriminator.parameters():
                param.requires_grad = False
            for param in model.class_discriminator.parameters():
                param.requires_grad = False

            optimizer_ae.zero_grad()

            # 提供调整特征的参数
            class_sub = class_label

            # 前向传播
            #x_recon, z, batch_pred, class_pred, sigmatrix, intermediate_result = model(x, expected_sigmatrix, class_sub, proportions_per_feature)
            x_recon, z, batch_pred, class_pred, sigmatrix, intermediate_result = model(x, expected_sigmatrix, class_sub, proportions_per_feature, calculate_intermediate=True)

            # 检查形状是否匹配
            #print(f"sigmatrix shape: {sigmatrix.shape}")
            #print(f"expected_sigmatrix shape: {expected_sigmatrix.shape}")
            #print(f"x_recon shape: {x_recon.shape}")   
            #print(f"x shape: {x.shape}")              
            #print(f"z shape: {z.shape}")         
            #print(f"batch_pred shape: {batch_pred.shape}")    
            #print(f"batch_label shape: {batch_label.shape}")                
            #print(f"class_pred shape: {class_pred.shape}")      
            #print(f"class_label shape: {class_label.shape}")                
            #print(f"intermediate_result shape: {intermediate_result.shape}")                     
            
            # 重构损失
            recon_loss = criterion_recon(x_recon, x)
            
            #与比例计算得到的损失
            #recon_loss2 = criterion_recon(x_recon, x_recon2)           
            #sigmatrix_loss_all.append(recon_loss2.item())

            # 类别判别器损失
            class_disc_loss_ae = criterion_disc(class_pred, class_label)

            # 批次判别器损失
            batch_disc_loss_ae = criterion_disc(batch_pred, batch_label)
            
            # Sigmatrix 损失   
            sigmatrix_loss = criterion_sigmatrix(sigmatrix, expected_sigmatrix)
            sigmatrix_loss_all.append(sigmatrix_loss.item())

            # Intermediate result 损失
            intermediate_loss = 0
            # 遍历每个样本
            for i in range(intermediate_result.size(0)):  # batch_size 次循环

                # 当前样本的所有特征
                current_features = intermediate_result[i]  # (feature_dim, num_genes)

                # 计算特征两两差异的平方，保留所有 pair-wise 差异的均值
                feature_differences = torch.cdist(current_features, current_features, p=2)  # (feature_dim, feature_dim)
                feature_mse = torch.mean(feature_differences ** 2)  # 平均差异

                # 累加当前样本的损失
                intermediate_loss += feature_mse

                # 对 batch_size 求平均
                intermediate_loss /= intermediate_result.size(0)            
            
            
            #for i in range(intermediate_result.size(0)):
            #    feature_means = intermediate_result[i].mean(dim=1)
                # 检查 intermediate_result 和 feature_means 的形状是否匹配           
            #    print(f"intermediate_result shape: {intermediate_result[i].shape}")
            #    print(f"feature_means shape: {feature_means.unsqueeze(1).shape}")
            #    intermediate_loss += criterion_intermediate(intermediate_result[i], feature_means.unsqueeze(1))
            #intermediate_loss /= intermediate_result.size(0)
            intermediate_loss_all.append(intermediate_loss.item())
            
            # 计算类中心损失
            class_center_loss = 0.0
            for class_id in class_label.unique():
                # 获取当前类的样本
                class_mask = (class_label == class_id)
                class_samples = z[class_mask]  # 当前类的样本特征 (num_samples_in_class, feature_dim)

                # 当前批次的类均值
                current_class_mean = class_samples.mean(dim=0)

                if class_id.item() not in class_centers:
                    # 如果类中心尚未初始化，使用当前批次均值初始化
                    class_centers[class_id.item()] = current_class_mean.detach()  # 分离计算图
                    class_sample_counts[class_id.item()] = class_samples.size(0)
                else:
                    # 如果类中心已初始化，使用加权平均更新
                    prev_center = class_centers[class_id.item()]
                    prev_count = class_sample_counts[class_id.item()]
                    total_count = prev_count + class_samples.size(0)

                    # 更新类中心
                    class_centers[class_id.item()] = (
                        (prev_center * prev_count + current_class_mean * class_samples.size(0)) / total_count
                    ).detach()  # 确保分离计算图
                    class_sample_counts[class_id.item()] = total_count

                # 计算当前类样本到类中心的距离，并累加到损失
                updated_center = class_centers[class_id.item()]
                class_center_loss += ((class_samples - updated_center.detach()) ** 2).sum()  # 分离计算图

            # 平均化类中心损失
            class_center_loss /= x.size(0)
            
            # 总损失
            total_loss = (
                1 * recon_loss +
                1 * class_disc_loss_ae +
                batch_disc_loss_ae +
                1 * sigmatrix_loss +
                class_center_loss +
                1 * intermediate_loss
            )

            total_loss.backward()
            optimizer_ae.step()

            recon_loss_all.append(recon_loss.item())

    return model, recon_loss_all, adv_loss, class_loss, sigmatrix_loss_all, intermediate_loss_all, class_centers

def train_model1(train_x, class_all, batch_labels, proportions_per_feature, expected_sigmatrix, 
                 model_name=None, batch_size=128, epochs=128):
    """
    训练自编码器模型。
    :param train_x: 输入特征数据，形状为 (样本数, 特征数)
    :param class_all: 每个样本的类别标签
    :param batch_labels: 每个样本的批次标签
    :param proportions_per_feature: 每个特征的调整比例，形状为 (feature_dim, num_classes, num_genes)
    :param expected_sigmatrix: 外部传入的 sigmatrix，形状为 (feature_dim, num_genes)
    :param model_name: 保存模型的文件名
    :param batch_size: 每个批次的大小
    :param epochs: 训练轮数
    :return: 训练好的模型
    """
    # 数据加载器
    train_loader = DataLoader(SimDataset1(train_x, class_all, batch_labels), 
                              batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    num_classes = len(np.unique(class_all))
    num_batches = len(np.unique(batch_labels))
    model = AutoEncoder(train_x.shape[1], 3, num_batches,num_classes).to(device)

    # 初始化优化器
    optimizer_ae = Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-4)
    optimizer_disc_batch = Adam(model.batch_discriminator.parameters(), lr=1e-4)
    optimizer_disc_class = Adam(model.class_discriminator.parameters(), lr=1e-4)

    print('Start training...')


    # 调用训练函数
    model, recon_loss_all, adv_loss, class_loss, sigmatrix_loss_all, intermediate_loss_all,class_centers = training_stage1(
        model, train_loader, optimizer_ae, optimizer_disc_batch, optimizer_disc_class,
        proportions_per_feature, expected_sigmatrix, epochs=epochs)
    

    # 打印损失
    print('Reconstruction loss:')
    showloss(recon_loss_all)
    print('Adversarial loss:')
    showloss(adv_loss)
    print('Class loss:')
    showloss(class_loss)
    print('Sigmatrix loss:')
    showloss(sigmatrix_loss_all)
    print('Intermediate result loss:')
    showloss(intermediate_loss_all)
    
    # 保存模型
    if model_name is not None:
        torch.save(model.state_dict(), model_name + ".pth")
    
    return model,class_centers

def run_logfc_analysis_and_generate_fc_array(
    sample_metadata,
    expr_matrix,
    output_path
):
    """
    从表型和表达矩阵中运行 shared category logFC 分析，并生成 fold-change 三维数组。
    
    参数:
        sample_metadata: DataFrame，含 disease 分类列
        expr_matrix: DataFrame，行为基因，列为样本
        output_path: str，输出结果文件夹
    
    返回:
        proportions_array_fc: np.ndarray, shape = (疾病数, 类别数, 基因数)
        top3_diseases: list，疾病名顺序
        fixed_categories: list，类别顺序
        fixed_genes: list，基因顺序
    """

    # Label 编码
    le = LabelEncoder()
    class_labels = le.fit_transform(sample_metadata['disease'])
    class_names = le.inverse_transform(class_labels)
    fixed_categories = list(le.classes_)

    # Top N 疾病类别
    top3_diseases = sample_metadata['disease'].value_counts().head(3).index.tolist()

    # 加载 R 脚本并获取函数
    # 使用 importlib.resources 获取 R 脚本的路径
    with resources.path('deepadvancer', 'shared_logFC_analysis.R') as r_script_path:
        r.source(str(r_script_path))
    #robjects.r['source']('/media/lab_chen/7a17f691-f95e-41bc-93b9-865a0241ff7a/CMT/Multi-agent/test/Basic_code/shared_logFC_analysis.R')
    run_logFC = robjects.globalenv['run_shared_category_logFC_analysis']

    # 转换表型和表达矩阵为 R 对象
    with localconverter(default_converter + pandas2ri.converter):
        r_phenotype = pandas2ri.py2rpy(sample_metadata)
        r_expression = pandas2ri.py2rpy(expr_matrix)

    # 运行分析
    for disease in top3_diseases:
        print(f"🔍 正在分析: {disease} ...")
        result = run_logFC(r_phenotype, r_expression, disease, output_path, 0.3)
        output_file = result.rx2('output_file')[0]
        print(f"✅ 保存至: {output_file}")
 
    def clean_disease_name(name):
        return name.replace('.', '_').replace('-', '_')
        
    # 构建 {清理后名称: 原始名称} 的反查表
    reverse_name_map = {clean_disease_name(d): d for d in top3_diseases}
        
    merged_results_dict = {}
    file_list = glob.glob(os.path.join(output_path, '*_merged_results.csv'))

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        cleaned_name = file_name.replace('_merged_results.csv', '')
        original_name = reverse_name_map[cleaned_name]

        try:
            df = pd.read_csv(file_path, index_col=0)
            merged_results_dict[original_name] = df
            print(f"✅ 成功读取: {file_name} -> 映射为: {original_name}")
        except Exception as e:
            print(f"❌ 读取失败: {file_name}，错误信息: {e}")


    # 基准对齐
    fixed_categories = list(le.classes_) 
    fixed_genes = list(expr_matrix.index)      

    def logfc_to_fc_clipped(logfc_df, clip_val=5):
        logfc_clipped = logfc_df.clip(lower=-clip_val, upper=clip_val)
        return 2 ** logfc_clipped

    # 初始化 3D 数组
    num_diseases = len(top3_diseases)
    num_categories = len(fixed_categories)
    num_genes = len(fixed_genes)
    proportions_array_fc = np.empty((num_diseases, num_categories, num_genes))

    for i, disease in enumerate(top3_diseases):
        df_logfc = merged_results_dict[disease]
        df_logfc_aligned = df_logfc.reindex(index=fixed_genes, columns=fixed_categories)
        df_fc = logfc_to_fc_clipped(df_logfc_aligned)
        proportions_array_fc[i] = df_fc.T.values  # 类别 × 基因

    proportions_array_fc = np.nan_to_num(proportions_array_fc, nan=1)

    print("✅ proportions_array_fc.shape =", proportions_array_fc.shape)
    
    

    # 加载 R 脚本，确保其中定义了 process_batch_correction1 函数
    #robjects.r['source']('/media/lab_chen/7a17f691-f95e-41bc-93b9-865a0241ff7a/CMT/Multi-agent/test/Basic_code/process_batch_correction.R')
    with resources.path('deepadvancer', 'process_batch_correction.R') as r_script_path:
        r.source(str(r_script_path))
    # 获取函数引用
    process_batch = robjects.globalenv['process_batch_correction1']


    # 转换为 R 对象
    with localconverter(default_converter + pandas2ri.converter):
        r_pheno = pandas2ri.py2rpy(sample_metadata )
        r_expr = pandas2ri.py2rpy(expr_matrix)

    # 设置参数
    target_disease = top3_diseases[0]
    shared_disease = top3_diseases[1]
    secondary_disease = top3_diseases[2]

    # 调用 R 函数
    result = process_batch(target_disease, shared_disease, secondary_disease, r_pheno, r_expr)

    # 提取返回的 R 对象中的 3 个部分
    subset_expr = result.rx2('subset_expression')
    corrected_expr = result.rx2('corrected_expression')
    subset_meta = result.rx2('subset_data')

    # 转回 pandas
    with localconverter(default_converter + pandas2ri.converter):
        df_expr = pandas2ri.rpy2py(corrected_expr)
        df_meta = pandas2ri.rpy2py(subset_meta)

    # 确保列名对齐
    df_expr = pd.DataFrame(df_expr, columns=df_meta['geo_accession'].values, index=expr_matrix.index)


    # 获取三个类别样本
    samples1 = df_meta[df_meta['disease'] == top3_diseases[0]]['geo_accession']
    samples2 = df_meta[df_meta['disease'] == top3_diseases[1]]['geo_accession']
    samples3 = df_meta[df_meta['disease'] == top3_diseases[2]]['geo_accession']

    # 提取子矩阵
    expr1 = df_expr[samples1]
    expr2 = df_expr[samples2]
    expr3 = df_expr[samples3]

    # 计算中心表达矩阵
    center_exp = pd.DataFrame({
        top3_diseases[0]: expr1.mean(axis=1),
        top3_diseases[1]: expr2.mean(axis=1),
        top3_diseases[2]: expr3.mean(axis=1)
    }).T


    # 创建字典存储 source_dataset 对应的 disease
    dataset_disease_mapping = defaultdict(set)

    # 遍历每一行，填充字典
    for _, row in sample_metadata.iterrows():
        dataset_disease_mapping[row['source_dataset']].add(row['disease'])

    # 构建每个 top3_diseases 相关联的 disease 集合
    related_diseases_dict = {}

    for disease in top3_diseases:
        datasets_with_disease = {
            dataset: diseases for dataset, diseases in dataset_disease_mapping.items()
            if disease in diseases
        }
    
        related = set()
        for dataset, diseases in datasets_with_disease.items():
            related.update(diseases - {disease})
    
        related_diseases_dict[disease] = related
        #print(f"Diseases that appear with '{disease}' in the same dataset:")
        #print(", ".join(sorted(related)))



    # 提取对应集合
    rel_healthy = related_diseases_dict[top3_diseases[0]]
    rel_psoriasis = related_diseases_dict[top3_diseases[1]]
    rel_ad = related_diseases_dict[top3_diseases[2]]

    # 交集运算
    three_class_shared = rel_healthy & rel_psoriasis & rel_ad
    three_class_shared.update(top3_diseases)


    feature1_feature2_shared = rel_healthy & rel_psoriasis
    feature1_feature3_shared = rel_healthy & rel_ad
    feature2_feature3_shared = rel_psoriasis & rel_ad




    # 调整行顺序和列顺序
    df_exp_aligned = center_exp.reindex(index=top3_diseases, columns=fixed_genes)


    # 获取整个 DataFrame 的全局最小值与最大值
    global_min = df_exp_aligned.values.min()
    global_max = df_exp_aligned.values.max()

    # 标准化计算：每个元素减去最小值再除以范围
    sigmatrix = (df_exp_aligned - global_min) / (global_max - global_min)
    sigmatrix


    # 初始化
    max_iterations = 100
    initial_learning_rate = 0.001
    final_learning_rate = 0.001
    initial_regularization_weight = 0.0001
    final_regularization_weight = 0.001

    # 将 DataFrame 转为 numpy，防止索引错误
    sigmatrix_new = sigmatrix.values.copy()
    sigmatrix_orig = sigmatrix.values.copy()  # 用于正则项和融合

    # 获取类别索引
    def get_class_indices(shared_set):
        return [
            np.where(le.classes_ == cls)[0][0]
            for cls in shared_set if cls in le.classes_
        ]

    three_class_shared_indices = get_class_indices(three_class_shared)
    feature1_feature2_shared_indices = get_class_indices(feature1_feature2_shared)
    feature1_feature3_shared_indices = get_class_indices(feature1_feature3_shared)
    feature2_feature3_shared_indices = get_class_indices(feature2_feature3_shared)

    # 优化 sigmatrix
    for iteration in range(max_iterations):
        # 动态调整学习率和正则化权重
        learning_rate = initial_learning_rate - (initial_learning_rate - final_learning_rate) * (iteration / max_iterations)
        regularization_weight = initial_regularization_weight + (
            (final_regularization_weight - initial_regularization_weight) * iteration / max_iterations
        )

        total_loss = 0
        correlation_loss = 0

        # 三个特征同时共现的类别
        for target_cls_index in three_class_shared_indices:
            values = np.array([
                sigmatrix_new[cls_idx, :] * proportions_array_fc[cls_idx, target_cls_index, :]
                for cls_idx in range(sigmatrix_new.shape[0])
            ])  # shape: (3, 12754)

            # Pearson 相关性计算（包含 NaN 防护）
            corr_01 = np.nan_to_num(pearsonr(values[0], values[1])[0])
            corr_02 = np.nan_to_num(pearsonr(values[0], values[2])[0])
            corr_12 = np.nan_to_num(pearsonr(values[1], values[2])[0])

            loss_corr = -(corr_01 + corr_02 + corr_12)
            correlation_loss += loss_corr

            # 梯度估算（中心化差异）
            mean_values = values.mean(axis=0)
            grad = np.stack([v - mean_values for v in values])
            grad += regularization_weight * (sigmatrix_new - sigmatrix_orig)

            # 随机扰动
            noise = np.random.normal(0, 0.0001, size=sigmatrix_new.shape)
            sigmatrix_new -= learning_rate * grad + noise

        # 特征 1 和 2
        for target_cls_index in feature1_feature2_shared_indices:
            values = np.array([
                sigmatrix_new[0, :] * proportions_array_fc[0, target_cls_index, :],
                sigmatrix_new[1, :] * proportions_array_fc[1, target_cls_index, :]
            ])
            corr = np.nan_to_num(pearsonr(values[0], values[1])[0])
            loss_corr = -corr
            correlation_loss += loss_corr

            mean_values = values.mean(axis=0)
            grad = np.stack([v - mean_values for v in values])
            grad += regularization_weight * (sigmatrix_new[[0, 1], :] - sigmatrix_orig[[0, 1], :])
            sigmatrix_new[[0, 1], :] -= learning_rate * grad

        # 特征 1 和 3
        for target_cls_index in feature1_feature3_shared_indices:
            values = np.array([
                sigmatrix_new[0, :] * proportions_array_fc[0, target_cls_index, :],
                sigmatrix_new[2, :] * proportions_array_fc[2, target_cls_index, :]
            ])
            corr = np.nan_to_num(pearsonr(values[0], values[1])[0])
            loss_corr = -corr
            correlation_loss += loss_corr

            mean_values = values.mean(axis=0)
            grad = np.stack([v - mean_values for v in values])
            grad += regularization_weight * (sigmatrix_new[[0, 2], :] - sigmatrix_orig[[0, 2], :])
            sigmatrix_new[[0, 2], :] -= learning_rate * grad

        # 特征 2 和 3
        for target_cls_index in feature2_feature3_shared_indices:
            values = np.array([
                sigmatrix_new[1, :] * proportions_array_fc[1, target_cls_index, :],
                sigmatrix_new[2, :] * proportions_array_fc[2, target_cls_index, :]
            ])
            corr = np.nan_to_num(pearsonr(values[0], values[1])[0])
            loss_corr = -corr
            correlation_loss += loss_corr

            mean_values = values.mean(axis=0)
            grad = np.stack([v - mean_values for v in values])
            grad += regularization_weight * (sigmatrix_new[[1, 2], :] - sigmatrix_orig[[1, 2], :])
            sigmatrix_new[[1, 2], :] -= learning_rate * grad

        # --- 每轮迭代后统一做正则项 + 平滑融合 ---
        regularization_loss = regularization_weight * np.linalg.norm(sigmatrix_new - sigmatrix_orig)
        total_loss = correlation_loss + regularization_loss

        # 非负裁剪
        sigmatrix_new = np.clip(sigmatrix_new, 0, None)

        # 融合原始值（平滑）
        sigmatrix_new = 0.9 * sigmatrix_new + 0.1 * sigmatrix_orig
        print(f"[Iter {iteration+1}] Corr Loss: {correlation_loss:.4f}, Reg Loss: {regularization_loss:.4f}, Total: {total_loss:.4f}")

    
    
    # 保存结果
    sigmatrix_df = pd.DataFrame(sigmatrix_new, index=top3_diseases, columns=fixed_genes)
    sigmatrix_df.to_csv(os.path.join(output_path, 'sigmatrix.csv'))
    print("✅ sigmatrix计算完成")
    
    
    
    # 按数据集进行 min-max 归一化
    normalized_datasets = []    
    for dataset_name, group in sample_metadata.groupby('source_dataset'):
        sample_ids = group.index.tolist()
        valid_samples = [s for s in sample_ids if s in expr_matrix.columns]
        if not valid_samples:
            continue
        sub_expr = expr_matrix[valid_samples].copy()
        dataset_max = sub_expr.values.max()
        dataset_min = sub_expr.values.min()
        if dataset_max == dataset_min:
            print(f"⚠️ 丢弃无信息数据集: {dataset_name}")
            continue
        sub_expr_scaled = (sub_expr - dataset_min) / (dataset_max - dataset_min)
        normalized_datasets.append(sub_expr_scaled)

    final_expression_matrix_scaled = pd.concat(normalized_datasets, axis=1)
    final_expression_matrix_scaled

    train_x = final_expression_matrix_scaled.values.T.astype(np.float32)


    class_all = le.fit_transform(sample_metadata["disease"])
    batch_le = LabelEncoder()
    batch_labels = batch_le.fit_transform(sample_metadata["source_dataset"])
    proportions_per_feature = torch.tensor(proportions_array_fc, dtype=torch.float32).to(device)
    expected_sigmatrix = torch.tensor(sigmatrix_new, dtype=torch.float32).to(device)



    return  train_x, class_all, batch_labels, proportions_per_feature, expected_sigmatrix

def recon_training(
    train_x=train_x,
    class_all=class_all,
    batch_labels=batch_labels,
    proportions_per_feature=proportions_per_feature,
    expected_sigmatrix=expected_sigmatrix,
    output_path,
    batch_size=128,
    epochs=300
):   
    model, class_centers = train_model1(
        train_x=train_x,
        class_all=class_all,
        batch_labels=batch_labels,
        proportions_per_feature=proportions_per_feature,
        expected_sigmatrix=expected_sigmatrix,
        model_name=os.path.join(output_path, 'mymodel'),  # 可选
        batch_size=batch_size,
        epochs=epochs)

    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(train_x).float().to(device)
        class_tensor = torch.tensor(class_all).long().to(device)

       # 推理调用
        x_recon, _, _, _, _, _ = model(
            x_tensor,
            expected_sigmatrix=expected_sigmatrix,
            class_sub=class_tensor,
            proportions_per_feature=proportions_per_feature,
            calculate_intermediate=True # 设置为 False 表示不返回中间结果
        )

    # 转为 numpy 以便保存/分析
    x_recon_np = x_recon.cpu().numpy()
    # 转置 reconstructed 矩阵，使其与 expr_matrix_filtered 的结构一致：基因在行，样本在列
    x_recon_expression_matrix = pd.DataFrame(
        x_recon_np.T,  # 转置，使形状变成 (基因数, 样本数)
        index=expr_matrix.index,  # 基因名
        columns=expr_matrix.columns  # 样本名
    )    
    return  x_recon_expression_matrix, model   

def compute_logfc_between_classes(
    expression_matrix: pd.DataFrame,
    phenotype_metadata: pd.DataFrame,
    class_column: str,
    class1: str,
    class2: str
) -> pd.DataFrame:
    """
    计算两个类之间的 log2 fold change（logFC）。
    
    参数：
        expression_matrix: DataFrame，行为基因，列为样本
        phenotype_metadata: DataFrame，索引为样本名，包含类别信息
        class_column: str，类别所在的列名
        class1: str，第一个类名（作为分母）
        class2: str，第二个类名（作为分子）
        
    返回：
        DataFrame，包含 log2FC 列，索引为基因名
    """

    # 获取两个类的样本索引
    samples1 = phenotype_metadata[phenotype_metadata[class_column] == class1].index
    samples2 = phenotype_metadata[phenotype_metadata[class_column] == class2].index

    # 提取表达子矩阵
    data1 = expression_matrix[samples1]
    data2 = expression_matrix[samples2]

    # 计算平均表达
    mean1 = data1.mean(axis=1)
    mean2 = data2.mean(axis=1)

    # 计算 log2 fold change
    logfc = np.log2(mean2 + 1e-8) - np.log2(mean1 + 1e-8)

    # 封装为 DataFrame
    result = pd.DataFrame({f"log2FC_{class2}_vs_{class1}": logfc})
    return result.sort_values(by=result.columns[0], ascending=False)

def compute_logfc_vs_others(
    expression_matrix: pd.DataFrame,
    phenotype_metadata: pd.DataFrame,
    class_column: str,
    target_class: str
) -> pd.DataFrame:
    """
    计算目标类 vs 所有其他类的 log2 fold change。
    
    参数：
        expression_matrix: DataFrame，行为基因，列为样本
        phenotype_metadata: DataFrame，索引为样本名，包含类别信息
        class_column: str，类别所在的列名
        target_class: str，要比较的类名
        
    返回：
        DataFrame，index 为基因，columns 为其他类别名，每列是 target_class vs other_class 的 log2FC
    """
    # 提取目标类样本并计算平均表达
    target_samples = phenotype_metadata[phenotype_metadata[class_column] == target_class].index
    target_data = expression_matrix[target_samples]
    mean_target = target_data.mean(axis=1)

    # 初始化结果 DataFrame
    logfc_df = pd.DataFrame(index=expression_matrix.index)

    # 遍历所有其他类
    for other_class in phenotype_metadata[class_column].unique():
        if other_class == target_class:
            continue
        other_samples = phenotype_metadata[phenotype_metadata[class_column] == other_class].index
        other_data = expression_matrix[other_samples]
        mean_other = other_data.mean(axis=1)

        # 计算 logFC
        logfc = np.log2(mean_target + 1e-8) - np.log2(mean_other + 1e-8)
        logfc_df[f'{target_class}_vs_{other_class}'] = logfc

    return logfc_df

