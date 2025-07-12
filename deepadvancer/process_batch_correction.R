
process_batch_correction1 <- function(target_diseases, shared_diseases, secondary_shared_diseases, phenotype_data, expression_data) {
  # Combine target and shared diseases
  all_diseases <- c(target_diseases, shared_diseases, secondary_shared_diseases)
  
  # Identify valid datasets with at least two of the four categories
  valid_datasets <- phenotype_data %>%
    filter(disease %in% all_diseases) %>%
    group_by(source_dataset) %>%
    summarize(num_categories = n_distinct(disease)) %>%
    filter(num_categories >= 2) %>%
    pull(source_dataset)
  
  if (length(valid_datasets) == 0) {
    stop("没有符合条件的 valid_datasets，请检查输入数据或参数")
  }
  
  # Initialize corrected expression matrix
  corrected_expression_list <- list()
  
  # Initialize merged expression matrix
  merged_log_transformed_expression <- NULL
  
  # Process each valid dataset
  for (dataset_id in valid_datasets) {
    #message(paste("Processing dataset:", dataset_id))
    
    
    # Select samples for the dataset
    selected_samples <- phenotype_data$source_dataset == dataset_id & phenotype_data$disease %in% all_diseases
    if (sum(selected_samples) <= 2) {
      message(paste("Dataset", dataset_id, "样本不足，跳过"))
      next
    }    
    
    expression_filtered <- expression_data[, selected_samples]
  
    # 判断数据集中是否有行在所有样本中都相同
    identical_rows <- apply(expression_filtered, 1, function(row) all(row == row[1], na.rm = TRUE))
    
    if (any(identical_rows)) {
      #message(paste("Dataset", dataset_id, "contains rows with identical values. Replacing them with median values."))
      # 替换相同行的表达值为中位数
      # median_values <- apply(expression_filtered[identical_rows, , drop = FALSE], 1, median, na.rm = TRUE)
      for (i in which(identical_rows)) {
        expression_filtered[i, ] <- 0
      }
    }    
      
    # Calculate quantiles
    qx <- as.numeric(quantile(expression_filtered, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm = TRUE))
    
    # Determine whether log2 transformation is needed
    LogC <- (qx[5] > 100) || (qx[6] - qx[1] > 50 && qx[2] > 0)
    
    if (LogC) {
      #message("Performing log2 transformation on the expression data.")
      expression_filtered[expression_filtered == 0] <- NaN  # Replace non-positive values
      expression_filtered <- log2(expression_filtered)
      expression_filtered[is.nan(as.matrix(expression_filtered))] <- 0   
    } #else {
    #message("Log2 transformation not required.")
    #}
    
    
    # Merge expression data
    if (is.null(merged_log_transformed_expression)) {
      merged_log_transformed_expression <- expression_filtered
    } else {
      merged_log_transformed_expression <- cbind(merged_log_transformed_expression, expression_filtered)
    }
  }
  
  # Filter samples with valid datasets
  subset_data <- phenotype_data[phenotype_data$source_dataset %in% valid_datasets, ]
  
  # Filter samples with target diseases
  subset_data <- subset_data[subset_data$disease %in% all_diseases, ]
  subset_expression <- merged_log_transformed_expression[, subset_data$geo_accession]
  
  # Perform batch effect correction
  batch <- factor(subset_data$source_dataset)
  mod <- model.matrix(~ disease, data = subset_data)
  
  corrected_expression <- removeBatchEffect(subset_expression, batch = batch, design = mod)
  
  # Return corrected expression matrix
  return(list(
    subset_expression = subset_expression,
    corrected_expression = corrected_expression,
    subset_data = subset_data
    
  ))
}


