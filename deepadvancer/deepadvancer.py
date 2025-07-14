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
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_process_expression_data(
    data_dir,
    output_dir=None,
    gene_threshold_ratio=0.8,
    dataset_gene_coverage_threshold=0.8
):
    """
    Load and integrate multiple expression matrices and phenotype files from a directory,
    followed by cleaning and normalization.

    Parameters:
        data_dir: str  
            Path to the main directory. Each subfolder should contain one expression matrix 
            file (with "expression" in its name) and one phenotype file (with "phenotype" 
            in its name), both in .csv format.
        output_dir: str or None  
            If specified, the processed results will be saved to this directory.
        gene_threshold_ratio: float  
            Threshold to define "shared genes" across datasets (default: present in at least 80% of datasets).
        dataset_gene_coverage_threshold: float  
            Minimum proportion of shared genes required for a dataset to be retained (default: 80%).

    Returns:
        final_expression_matrix_scaled: pd.DataFrame  
            The normalized and merged expression matrix.
        phenotype_metadat_scaled: pd.DataFrame  
            The corresponding cleaned and merged sample phenotype metadata.
    """


    expression_data = {}
    phenotype_data = {}

    # 读取所有表达和表型文件
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            expr_files = glob.glob(os.path.join(folder_path, '*expression*.csv'))
            pheno_files = glob.glob(os.path.join(folder_path, '*phenotype*.csv'))

            if expr_files and pheno_files:
                expr_file = expr_files[0]
                pheno_file = pheno_files[0]

                expr_df = pd.read_csv(expr_file, index_col=0)
                pheno_df = pd.read_csv(pheno_file, index_col=0)

                expression_data[folder_name] = expr_df
                phenotype_data[folder_name] = pheno_df

    print(f"{len(expression_data)} datasets loaded.")


    gene_sets = [set(df.index) for df in expression_data.values()]
    total_datasets = len(gene_sets)
    gene_counter = Counter(g for genes in gene_sets for g in genes)
    gene_threshold = int(total_datasets * gene_threshold_ratio)
    common_genes = [gene for gene, count in gene_counter.items() if count >= gene_threshold]
    print(f"Number of common genes (present in ≥{gene_threshold_ratio:.0%} of datasets): {len(common_genes)}")


    filtered_expression_data = {}
    filtered_phenotype_data = {}

    for dataset_name, expr_df in expression_data.items():
        gene_overlap = set(common_genes).intersection(expr_df.index)
        overlap_ratio = len(gene_overlap) / len(common_genes)

        if overlap_ratio >= dataset_gene_coverage_threshold:
            filtered_expr_df = expr_df.loc[list(gene_overlap)].copy()
            filtered_expression_data[dataset_name] = filtered_expr_df
            filtered_phenotype_data[dataset_name] = phenotype_data[dataset_name]
            print(f"✅ Retained dataset: {dataset_name} (shared gene ratio: {overlap_ratio:.2%})")
        else:
            print(f"❌ Discarded dataset: {dataset_name} (shared gene ratio: {overlap_ratio:.2%})")


    aligned_dfs = [df.reindex(common_genes) for df in filtered_expression_data.values()]


    expression_combined = pd.concat(aligned_dfs, axis=1)
    expression_combined = expression_combined.apply(lambda row: row.fillna(row.mean()), axis=1)
   
    


    all_pheno = []
    for dataset_name, df in filtered_phenotype_data.items():
        df = df.copy()
        df['source_dataset'] = dataset_name
        all_pheno.append(df)

    phenotype_combined = pd.concat(all_pheno, axis=0)


    expression_combined = expression_combined.loc[:, ~expression_combined.columns.duplicated()]
    phenotype_combined = phenotype_combined.loc[~phenotype_combined.index.duplicated(keep='first')]


    shared_samples = expression_combined.columns.intersection(phenotype_combined.index)
    expression_combined = expression_combined[shared_samples]
    phenotype_combined = phenotype_combined.loc[shared_samples]

    sample_metadat_cleaned = phenotype_combined
    expression_matrix_cleaned = expression_combined


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


    final_expression_matrix = pd.concat(processed_datasets, axis=1)
    print("All datasets processed and combined.")

    rows_to_remove = []
    for row in final_expression_matrix.index:
        value_counts = final_expression_matrix.loc[row].value_counts(normalize=True)
        if value_counts.max() >= 0.6:
            rows_to_remove.append(row)

    final_expression_matrix = final_expression_matrix.drop(index=rows_to_remove)
    #sample_metadat_cleaned = sample_metadat_cleaned.drop(index=rows_to_remove)
    print(f"Removed {len(rows_to_remove)} rows with low expression variance.")

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
            print(f"⚠️ Discarded non-informative dataset: {dataset_name}")
            continue
        sub_expr_scaled = sub_expr#(sub_expr - dataset_min) / (dataset_max - dataset_min)
        normalized_datasets.append(sub_expr_scaled)

    final_expression_matrix_scaled = pd.concat(normalized_datasets, axis=1)


    retained_samples = final_expression_matrix_scaled.columns
    phenotype_metadat_scaled = sample_metadat_cleaned.loc[
        sample_metadat_cleaned.index.intersection(retained_samples)
    ]
    phenotype_metadat_scaled = phenotype_metadat_scaled.loc[retained_samples]

    print(f"Normalized expression matrix shape: {final_expression_matrix_scaled.shape}")
    print(f"Normalized phenotype metadata shape: {phenotype_metadat_scaled.shape}")

    final_expression_matrix_scaled.to_csv(os.path.join(output_dir, "expression_matrix_scaled.csv"))
    phenotype_metadat_scaled.to_csv(os.path.join(output_dir, "phenotype_metadat_scaled.csv"))
    print(f"✅ Successfully saved: file written to {output_dir}")


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
    Compute gene expression values based on the adjusted latent representation (Z) 
    and the provided sigmatrix.

    Parameters:
        adjusted_z: torch.Tensor  
            Adjusted latent features with shape (batch_size, feature_dim, num_genes).
        sigmatrix: torch.Tensor  
            Signal matrix with shape (feature_dim, num_genes).
        class_labels: torch.Tensor  
            Class labels for each sample in the batch, shape (batch_size,).
        calculate_intermediate: bool, default=True  
            Whether to compute and return intermediate results.

    Returns:
        expression_output: torch.Tensor  
            Reconstructed gene expression values with shape (batch_size, num_genes).
        intermediate_result: dict (optional)  
            Intermediate values used during computation (returned only if calculate_intermediate is True).
    """

    batch_size, feature_dim, num_genes = adjusted_z.size()


    gene_expression = torch.zeros((batch_size, num_genes), device=adjusted_z.device)

    if calculate_intermediate:
        intermediate_result = torch.zeros((batch_size, feature_dim, num_genes), device=adjusted_z.device)

    for i in range(batch_size):

        current_adjusted_z = adjusted_z[i]  # (feature_dim, num_genes)
        sample_class = class_labels[i].item()  
        if calculate_intermediate:
            for j in range(feature_dim):

                intermediate_result[i, j] = current_adjusted_z[j] * sigmatrix[j]  # (num_genes,)

            gene_expression[i] = intermediate_result[i].sum(dim=0) + 3 * intermediate_result[i, 0] # + 0.1 * b4 # 特征1比例变为 3 倍
        else:
            gene_expression[i] = torch.matmul(current_adjusted_z, sigmatrix.T).sum(dim=0) #+ b4 
            

    if calculate_intermediate:
        return gene_expression, intermediate_result
    else:
        return gene_expression 
    
def adjust_features(z, class_sub, proportions_per_feature):
    """
    Construct and apply adjustment matrices for each feature based on class_sub and 
    proportions_per_feature, then apply them to the latent representation z.

    Parameters:
        z: torch.Tensor  
            Encoded latent features with shape (batch_size, feature_dim).
        class_sub: torch.Tensor  
            Class labels for the current batch, shape (batch_size,).
        proportions_per_feature: torch.Tensor  
            Adjustment ratios per feature, tensor of shape (feature_dim, num_classes, num_genes).

    Returns:
        torch.Tensor  
            Adjusted feature tensor with shape (batch_size, feature_dim, num_genes).
    """


    proportions_per_feature = proportions_per_feature.to(z.device)
    

    batch_size, feature_dim = z.size()  # (batch_size, feature_dim)
    num_classes, num_genes = proportions_per_feature.size(1), proportions_per_feature.size(2)

    class_one_hot = F.one_hot(class_sub, num_classes=num_classes).float()  # (batch_size, num_classes)
    

    selected_proportions = torch.einsum('bc,fcg->bfg', class_one_hot, proportions_per_feature)


    z_expanded = z.unsqueeze(-1)  # (batch_size, feature_dim, 1)
    

    adjusted_features = z_expanded * selected_proportions  # (batch_size, feature_dim, num_genes)


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


    def sigmatrix_with_bias(self):
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


        w01 = torch.mm(w0, w1)  # 形状: (z_dim, 128)
        #out01 = w01 + b1.unsqueeze(0)  

        w02 = torch.mm(w01, w2)  # 形状: (z_dim, 256)
        #out02 = w02 + b2.unsqueeze(0)  

        w03 = torch.mm(w02, w3)  # 形状: (z_dim, 512)
        #out03 = w03 + b3.unsqueeze(0)  

        w04 = torch.mm(w03, w4)  # 形状: (z_dim, inputdim)
        #out04 = w04 + b4.unsqueeze(0)  

        return w04 


    def forward(self, x, expected_sigmatrix, class_sub, proportions_per_feature, calculate_intermediate):
        sigmatrix = self.sigmatrix_with_bias()
        #print(b04.shape)
        z = self.encode(x)
        #x_recon1 = self.decode(z)

        # Predict batch and class
        batch_pred = self.batch_discriminator(z)
        class_pred = self.class_discriminator(z)

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
        self.X = X  
        self.class_labels_all = class_labels_all  
        self.batch_labels_all = batch_labels_all  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        class_label_all = torch.tensor(self.class_labels_all[index]).long()
        batch_label_all = torch.tensor(self.batch_labels_all[index]).long()

        return x.to(device), class_label_all.to(device), batch_label_all.to(device)
    
def training_stage1(model, train_loader, optimizer_ae, optimizer_disc_batch, optimizer_disc_class, 
                    proportions_per_feature, expected_sigmatrix, epochs=128):
    """
    training procedure.

    Parameters:
        model: torch.nn.Module  
            The AutoEncoder model.
        train_loader: DataLoader  
            Data loader for the training set.
        optimizer_ae: torch.optim.Optimizer  
            Optimizer for the autoencoder.
        optimizer_disc_batch: torch.optim.Optimizer  
            Optimizer for the batch discriminator.
        optimizer_disc_class: torch.optim.Optimizer  
            Optimizer for the class discriminator.
        proportions_per_feature: np.ndarray  
            Adjustment ratios for each feature, with shape (feature_dim, num_classes, num_genes).
        expected_sigmatrix: np.ndarray  
            Externally provided signal matrix, with shape (feature_dim, num_genes).
        epochs: int  
            Number of training epochs.

    Returns:
        model: torch.nn.Module  
            The trained model.
        loss_logs: dict  
            A dictionary containing training loss histories for different components.
    """

    model.train()
    model.state = 'train'

    criterion_recon = nn.MSELoss()
    criterion_disc = nn.CrossEntropyLoss()
    criterion_sigmatrix = nn.MSELoss()
    criterion_intermediate = nn.MSELoss()

    recon_loss_all = []
    adv_loss = []
    class_loss = []
    sigmatrix_loss_all = []
    intermediate_loss_all = []
  

    class_centers = {}  
    class_sample_counts = {} 

    for epoch in tqdm(range(epochs)):
        for x, class_label, batch_label in train_loader:
            
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

            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = True
            for param in model.batch_discriminator.parameters():
                param.requires_grad = False
            for param in model.class_discriminator.parameters():
                param.requires_grad = False

            optimizer_ae.zero_grad()


            class_sub = class_label


            #x_recon, z, batch_pred, class_pred, sigmatrix, intermediate_result = model(x, expected_sigmatrix, class_sub, proportions_per_feature)
            x_recon, z, batch_pred, class_pred, sigmatrix, intermediate_result = model(x, expected_sigmatrix, class_sub, proportions_per_feature, calculate_intermediate=True)


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
            

            recon_loss = criterion_recon(x_recon, x)
            

            class_disc_loss_ae = criterion_disc(class_pred, class_label)


            batch_disc_loss_ae = criterion_disc(batch_pred, batch_label)
            

            sigmatrix_loss = criterion_sigmatrix(sigmatrix, expected_sigmatrix)
            sigmatrix_loss_all.append(sigmatrix_loss.item())


            intermediate_loss = 0

            for i in range(intermediate_result.size(0)): 


                current_features = intermediate_result[i]  # (feature_dim, num_genes)


                feature_differences = torch.cdist(current_features, current_features, p=2)  # (feature_dim, feature_dim)
                feature_mse = torch.mean(feature_differences ** 2)  # 平均差异


                intermediate_loss += feature_mse

                intermediate_loss /= intermediate_result.size(0)            
            
            

            intermediate_loss_all.append(intermediate_loss.item())
            

            class_center_loss = 0.0
            for class_id in class_label.unique():
                class_mask = (class_label == class_id)
                class_samples = z[class_mask]  

                current_class_mean = class_samples.mean(dim=0)

                if class_id.item() not in class_centers:

                    class_centers[class_id.item()] = current_class_mean.detach() 
                    class_sample_counts[class_id.item()] = class_samples.size(0)
                else:
                    prev_center = class_centers[class_id.item()]
                    prev_count = class_sample_counts[class_id.item()]
                    total_count = prev_count + class_samples.size(0)

                    class_centers[class_id.item()] = (
                        (prev_center * prev_count + current_class_mean * class_samples.size(0)) / total_count
                    ).detach()  
                    class_sample_counts[class_id.item()] = total_count


                updated_center = class_centers[class_id.item()]
                class_center_loss += ((class_samples - updated_center.detach()) ** 2).sum()  # 分离计算图


            class_center_loss /= x.size(0)
            

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
    Train the autoencoder model.

    Parameters:
        train_x: np.ndarray  
            Input feature data with shape (num_samples, num_features).
        class_all: list or array-like  
            Class labels for each sample.
        batch_labels: list or array-like  
            Batch labels for each sample.
        proportions_per_feature: np.ndarray  
            Adjustment ratios for each feature, with shape (feature_dim, num_classes, num_genes).
        expected_sigmatrix: np.ndarray  
            Externally provided signal matrix with shape (feature_dim, num_genes).
        model_name: str  
            File name to save the trained model.
        batch_size: int  
            Number of samples per batch.
        epochs: int  
            Number of training epochs.

    Returns:
        model: torch.nn.Module  
            The trained autoencoder model.
    """


    train_loader = DataLoader(SimDataset1(train_x, class_all, batch_labels), 
                              batch_size=batch_size, shuffle=True)
    
    num_classes = len(np.unique(class_all))
    num_batches = len(np.unique(batch_labels))
    model = AutoEncoder(train_x.shape[1], 3, num_batches,num_classes).to(device)


    optimizer_ae = Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-4)
    optimizer_disc_batch = Adam(model.batch_discriminator.parameters(), lr=1e-4)
    optimizer_disc_class = Adam(model.class_discriminator.parameters(), lr=1e-4)

    print('Start training...')


    model, recon_loss_all, adv_loss, class_loss, sigmatrix_loss_all, intermediate_loss_all,class_centers = training_stage1(
        model, train_loader, optimizer_ae, optimizer_disc_batch, optimizer_disc_class,
        proportions_per_feature, expected_sigmatrix, epochs=epochs)
    

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

    if model_name is not None:
        torch.save(model.state_dict(), model_name + ".pth")
    
    return model,class_centers

def run_logfc_analysis_and_generate_fc_array(
    sample_metadata,
    expr_matrix,
    output_path
):
    """
    Run fold change analysis and generate the expected signal matrix (sigmatrix) 
    based on phenotype and expression data.

    Parameters:
        sample_metadata: DataFrame  
            Sample metadata containing at least a 'disease' column with class labels.
        expr_matrix: DataFrame  
            Gene expression matrix with genes as rows and samples as columns.
        output_path: str  
            Directory path to save the output files.

    Returns:
        train_x          
        class_all
        batch_labels
        proportions_per_feature
        expected_sigmatrix
    """

    le = LabelEncoder()
    class_labels = le.fit_transform(sample_metadata['disease'])
    class_names = le.inverse_transform(class_labels)
    fixed_categories = list(le.classes_)
    top3_diseases = sample_metadata['disease'].value_counts().head(3).index.tolist()

    with resources.path('deepadvancer', 'shared_logFC_analysis.R') as r_script_path:
        robjects.r['source'](str(r_script_path))

    run_logFC = robjects.globalenv['run_shared_category_logFC_analysis']


    with localconverter(default_converter + pandas2ri.converter):
        r_phenotype = pandas2ri.py2rpy(sample_metadata)
        r_expression = pandas2ri.py2rpy(expr_matrix)

    for disease in top3_diseases:
        print(f"🔍 Analyzing: {disease} ...")
        result = run_logFC(r_phenotype, r_expression, disease, output_path, 0.3)
        output_file = result.rx2('output_file')[0]
        print(f"✅ Saved to: {output_file}")

 
    def clean_disease_name(name):
        return name.replace('.', '_').replace('-', '_')
        
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
            print(f"✅ Successfully loaded: {file_name} -> mapped to: {original_name}")
        except Exception as e:
            print(f"❌ Failed to load: {file_name}, error: {e}")


    fixed_categories = list(le.classes_) 
    fixed_genes = list(expr_matrix.index)      

    def logfc_to_fc_clipped(logfc_df, clip_val=5):
        logfc_clipped = logfc_df.clip(lower=-clip_val, upper=clip_val)
        return 2 ** logfc_clipped

    num_diseases = len(top3_diseases)
    num_categories = len(fixed_categories)
    num_genes = len(fixed_genes)
    proportions_array_fc = np.empty((num_diseases, num_categories, num_genes))

    for i, disease in enumerate(top3_diseases):
        df_logfc = merged_results_dict[disease]
        df_logfc_aligned = df_logfc.reindex(index=fixed_genes, columns=fixed_categories)
        df_fc = logfc_to_fc_clipped(df_logfc_aligned)
        proportions_array_fc[i] = df_fc.T.values 

    proportions_array_fc = np.nan_to_num(proportions_array_fc, nan=1)

    print("✅ proportions_array_fc.shape =", proportions_array_fc.shape)
    
    

    #robjects.r['source']('/media/lab_chen/7a17f691-f95e-41bc-93b9-865a0241ff7a/CMT/Multi-agent/test/Basic_code/process_batch_correction.R')
    with resources.path('deepadvancer', 'process_batch_correction.R') as r_script_path:
        robjects.r['source'](str(r_script_path))
    process_batch = robjects.globalenv['process_batch_correction1']


    with localconverter(default_converter + pandas2ri.converter):
        r_pheno = pandas2ri.py2rpy(sample_metadata )
        r_expr = pandas2ri.py2rpy(expr_matrix)


    target_disease = top3_diseases[0]
    shared_disease = top3_diseases[1]
    secondary_disease = top3_diseases[2]


    result = process_batch(target_disease, shared_disease, secondary_disease, r_pheno, r_expr)
    subset_expr = result.rx2('subset_expression')
    corrected_expr = result.rx2('corrected_expression')
    subset_meta = result.rx2('subset_data')

    with localconverter(default_converter + pandas2ri.converter):
        df_expr = pandas2ri.rpy2py(corrected_expr)
        df_meta = pandas2ri.rpy2py(subset_meta)

    df_expr = pd.DataFrame(df_expr, columns=df_meta['geo_accession'].values, index=expr_matrix.index)


    samples1 = df_meta[df_meta['disease'] == top3_diseases[0]]['geo_accession']
    samples2 = df_meta[df_meta['disease'] == top3_diseases[1]]['geo_accession']
    samples3 = df_meta[df_meta['disease'] == top3_diseases[2]]['geo_accession']

    expr1 = df_expr[samples1]
    expr2 = df_expr[samples2]
    expr3 = df_expr[samples3]

    center_exp = pd.DataFrame({
        top3_diseases[0]: expr1.mean(axis=1),
        top3_diseases[1]: expr2.mean(axis=1),
        top3_diseases[2]: expr3.mean(axis=1)
    }).T


    dataset_disease_mapping = defaultdict(set)


    for _, row in sample_metadata.iterrows():
        dataset_disease_mapping[row['source_dataset']].add(row['disease'])


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


    rel_healthy = related_diseases_dict[top3_diseases[0]]
    rel_psoriasis = related_diseases_dict[top3_diseases[1]]
    rel_ad = related_diseases_dict[top3_diseases[2]]

    three_class_shared = rel_healthy & rel_psoriasis & rel_ad
    three_class_shared.update(top3_diseases)


    feature1_feature2_shared = rel_healthy & rel_psoriasis
    feature1_feature3_shared = rel_healthy & rel_ad
    feature2_feature3_shared = rel_psoriasis & rel_ad



    df_exp_aligned = center_exp.reindex(index=top3_diseases, columns=fixed_genes)


    global_min = df_exp_aligned.values.min()
    global_max = df_exp_aligned.values.max()

    sigmatrix = (df_exp_aligned - global_min) / (global_max - global_min)
    sigmatrix


    max_iterations = 100
    initial_learning_rate = 0.001
    final_learning_rate = 0.001
    initial_regularization_weight = 0.0001
    final_regularization_weight = 0.001

    sigmatrix_new = sigmatrix.values.copy()
    sigmatrix_orig = sigmatrix.values.copy()  

    def get_class_indices(shared_set):
        return [
            np.where(le.classes_ == cls)[0][0]
            for cls in shared_set if cls in le.classes_
        ]

    three_class_shared_indices = get_class_indices(three_class_shared)
    feature1_feature2_shared_indices = get_class_indices(feature1_feature2_shared)
    feature1_feature3_shared_indices = get_class_indices(feature1_feature3_shared)
    feature2_feature3_shared_indices = get_class_indices(feature2_feature3_shared)


    for iteration in range(max_iterations):
        learning_rate = initial_learning_rate - (initial_learning_rate - final_learning_rate) * (iteration / max_iterations)
        regularization_weight = initial_regularization_weight + (
            (final_regularization_weight - initial_regularization_weight) * iteration / max_iterations
        )

        total_loss = 0
        correlation_loss = 0

        for target_cls_index in three_class_shared_indices:
            values = np.array([
                sigmatrix_new[cls_idx, :] * proportions_array_fc[cls_idx, target_cls_index, :]
                for cls_idx in range(sigmatrix_new.shape[0])
            ])  

            corr_01 = np.nan_to_num(pearsonr(values[0], values[1])[0])
            corr_02 = np.nan_to_num(pearsonr(values[0], values[2])[0])
            corr_12 = np.nan_to_num(pearsonr(values[1], values[2])[0])

            loss_corr = -(corr_01 + corr_02 + corr_12)
            correlation_loss += loss_corr


            mean_values = values.mean(axis=0)
            grad = np.stack([v - mean_values for v in values])
            grad += regularization_weight * (sigmatrix_new - sigmatrix_orig)

            noise = np.random.normal(0, 0.0001, size=sigmatrix_new.shape)
            sigmatrix_new -= learning_rate * grad + noise


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

        regularization_loss = regularization_weight * np.linalg.norm(sigmatrix_new - sigmatrix_orig)
        total_loss = correlation_loss + regularization_loss

        sigmatrix_new = np.clip(sigmatrix_new, 0, None)


        sigmatrix_new = 0.9 * sigmatrix_new + 0.1 * sigmatrix_orig
        print(f"[Iter {iteration+1}] Corr Loss: {correlation_loss:.4f}, Reg Loss: {regularization_loss:.4f}, Total: {total_loss:.4f}")

    
    

    sigmatrix_df = pd.DataFrame(sigmatrix_new, index=top3_diseases, columns=fixed_genes)
    sigmatrix_df.to_csv(os.path.join(output_path, 'sigmatrix.csv'))
    print("✅ Sigmatrix computation completed")

    
    
    
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
            print(f"⚠️ Discarding non-informative dataset: {dataset_name}")
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
    expr_matrix,
    train_x,
    class_all,
    batch_labels,
    proportions_per_feature,
    expected_sigmatrix,
    output_path,
    batch_size=128,
    epochs=300
):   
    """
    Train the adversarial autoencoder to reconstruct expression data.

    Parameters:
        expr_matrix: DataFrame  
            Original gene expression matrix with genes as rows and samples as columns.
        train_x: np.ndarray  
            Fold change tensor generated from prior analysis, used as input for training.
        class_all: list  
            List of all unique class labels (e.g., disease types).
        batch_labels: list  
            List of batch labels corresponding to each sample.
        proportions_per_feature: np.ndarray  
            Matrix of expression proportions per gene and class, used for supervision.
        expected_sigmatrix: np.ndarray  
            Expected decoding matrix representing class-specific gene patterns.
        output_path: str  
            Directory where model checkpoints and results will be saved.
        batch_size: int  
            Number of samples per training batch (default: 128).
        epochs: int  
            Number of training epochs (default: 300).

    Returns:
        x_recon_expression_matrix: pd.DataFrame  
            The reconstructed and batch-corrected expression matrix.
        model: torch.nn.Module  
            The trained autoencoder model.
    """

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


        x_recon, _, _, _, _, _ = model(
            x_tensor,
            expected_sigmatrix=expected_sigmatrix,
            class_sub=class_tensor,
            proportions_per_feature=proportions_per_feature,
            calculate_intermediate=True 
        )

    x_recon_np = x_recon.cpu().numpy()

    x_recon_expression_matrix = pd.DataFrame(
        x_recon_np.T,  
        index=expr_matrix.index,  
        columns=expr_matrix.columns  
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
    Compute the log2 fold change (logFC) between two specified classes.

    Parameters:
        expression_matrix: DataFrame  
            Gene expression matrix with genes as rows and samples as columns.
        phenotype_metadata: DataFrame  
            Sample metadata with sample names as the index and class labels as columns.
        class_column: str  
            Name of the column in phenotype_metadata that contains class labels.
        class1: str  
            Name of the first class (used as the denominator in fold change calculation).
        class2: str  
            Name of the second class (used as the numerator in fold change calculation).

    Returns:
        DataFrame  
            A DataFrame with genes as index and a single column 'log2FC' representing 
            the log2 fold change of class2 versus class1.
    """


    samples1 = phenotype_metadata[phenotype_metadata[class_column] == class1].index
    samples2 = phenotype_metadata[phenotype_metadata[class_column] == class2].index
    data1 = expression_matrix[samples1]
    data2 = expression_matrix[samples2]
    mean1 = data1.mean(axis=1)
    mean2 = data2.mean(axis=1)
    logfc = np.log2(mean2 + 1e-8) - np.log2(mean1 + 1e-8)
    result = pd.DataFrame({f"log2FC_{class2}_vs_{class1}": logfc})
    return result.sort_values(by=result.columns[0], ascending=False)

def compute_logfc_vs_others(
    expression_matrix: pd.DataFrame,
    phenotype_metadata: pd.DataFrame,
    class_column: str,
    target_class: str
) -> pd.DataFrame:
    """
    Compute the log2 fold change of the target class versus all other classes.

    Parameters:
        expression_matrix: DataFrame  
            Gene expression matrix with genes as rows and samples as columns.
        phenotype_metadata: DataFrame  
            Sample metadata with sample names as the index and class labels as columns.
        class_column: str  
            Name of the column in phenotype_metadata that contains class labels.
        target_class: str  
            The class to be compared against all others.

    Returns:
        DataFrame  
            A log2 fold change matrix with genes as index and each column representing 
            the log2FC of target_class vs another class.
    """

    target_samples = phenotype_metadata[phenotype_metadata[class_column] == target_class].index
    target_data = expression_matrix[target_samples]
    mean_target = target_data.mean(axis=1)


    logfc_df = pd.DataFrame(index=expression_matrix.index)
    for other_class in phenotype_metadata[class_column].unique():
        if other_class == target_class:
            continue
        other_samples = phenotype_metadata[phenotype_metadata[class_column] == other_class].index
        other_data = expression_matrix[other_samples]
        mean_other = other_data.mean(axis=1)
        logfc = np.log2(mean_target + 1e-8) - np.log2(mean_other + 1e-8)
        logfc_df[f'{target_class}_vs_{other_class}'] = logfc

    return logfc_df   

def plot_pca_by_class(
    expression_df,
    class_series,
    jitter_scale=0.8,
    point_size=15,
    max_legend_classes=50,
    title="PCA of Expression Matrix",
    figsize=(12, 10),
    alpha=0.8,
    save_path=None
):
    """
    Perform PCA on the expression matrix and visualize samples by class. 
    Optionally, save the resulting plot.

    Parameters:
        expression_df: DataFrame  
            Gene expression matrix (genes x samples).
        class_series: pd.Series  
            Class labels aligned with the columns of the expression matrix.
        jitter_scale: float  
            Scale of coordinate jitter for better separation.
        point_size: int  
            Size of each scatter point in the plot.
        max_legend_classes: int  
            Maximum number of classes to show in the legend; legend is omitted if exceeded.
        title: str  
            Title of the PCA plot.
        figsize: tuple  
            Size of the figure (width, height).
        alpha: float  
            Transparency level of the points (0 to 1).
        save_path: str, optional  
            Path to save the PCA plot (e.g., "output/pca.png").
    """


    expression_df = expression_df.loc[:, class_series.index]
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(expression_df.T.values)
    le = LabelEncoder()
    class_labels = le.fit_transform(class_series)
    label_names = le.classes_
    n_classes = len(label_names)
    rng = np.random.default_rng(42)
    embedding_2d_jittered = embedding_2d + rng.normal(0, jitter_scale, size=embedding_2d.shape)
    cmap = cm.get_cmap("nipy_spectral", n_classes)
    plt.figure(figsize=figsize)
    for i in range(n_classes):
        idx = class_labels == i
        plt.scatter(
            embedding_2d_jittered[idx, 0],
            embedding_2d_jittered[idx, 1],
            s=point_size,
            color=cmap(i),
            label=label_names[i],
            alpha=alpha,
            edgecolors='none'
        )

    plt.title(title, fontsize=16)
    plt.xlabel("PCA Component 1", fontsize=14)
    plt.ylabel("PCA Component 2", fontsize=14)

    if n_classes <= max_legend_classes:
        plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        print(f"⚠️ Number of classes is {n_classes}; legend has been omitted.")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure saved to: {save_path}")

    plt.show()


    
    
    
