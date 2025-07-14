
library(dplyr)
library(limma)
process_differential_analysis <- function(target_diseases, shared_diseases, secondary_shared_diseases, phenotype_data, expression_data) {
  # Combine target and shared diseases target_diseases ="systemic sclerosis"
  all_diseases <- c(target_diseases, shared_diseases, secondary_shared_diseases)
  #all_diseases <- c("atopic dermatitis", "healthy", "scleroderma")
  # Identify valid datasets
  valid_datasets <- subset(phenotype_data, disease %in% all_diseases) %>%
    group_by(source_dataset) %>%
    summarize(num_categories = n_distinct(disease)) %>%
    filter(num_categories >= 2) %>%
    pull(source_dataset)
  
  # Initialize merged expression matrix
  merged_log_transformed_expression <- NULL
  
  # Process each valid dataset
  for (dataset_id in valid_datasets) {
    #message(paste("Processing dataset:", dataset_id))
    
    # Select samples for the dataset
    selected_samples <- phenotype_data$source_dataset == dataset_id
    expression_filtered <- expression_data[, selected_samples]

    identical_rows <- apply(expression_filtered, 1, function(row) all(row == row[1], na.rm = TRUE))
    if (any(identical_rows)) {
      #message(paste("Dataset", dataset_id, "contains rows with identical values. Replacing them with median values."))
      # median_values <- apply(expression_filtered[identical_rows, , drop = FALSE], 1, median, na.rm = TRUE)
      for (i in which(identical_rows)) {
        expression_filtered[i, ] <- NaN
      }
    }    

    # Calculate quantiles
    qx <- as.numeric(quantile(expression_filtered, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm = TRUE))
    
    # Determine whether log2 transformation is needed
    LogC <- (qx[5] > 100) || (qx[6] - qx[1] > 50 && qx[2] > 0)
    
    if (LogC) {
      #message("Performing log2 transformation on the expression data.")
      expression_filtered[expression_filtered <= 0] <- NaN  # Replace non-positive values
      expression_filtered <- log2(expression_filtered)
      expression_filtered[is.nan(as.matrix(expression_filtered))] <- 0    
    } #else {
      #expression_filtered[is.nan(as.matrix(expression_filtered))] <- 0  
      #message("Log2 transformation not required.")
    #}
    
    # Replace NA values with row means
    #expression_filtered <- as.data.frame(apply(expression_filtered, 1, function(row) {
    #  row_mean <- mean(row, na.rm = TRUE)
    #  row[is.na(row)] <- row_mean
   #   return(row)
   # }))
    
    expression_filtered <- as.data.frame(expression_filtered)  # Transpose back to original format
    
    #identical_rows <- apply(expression_filtered, 1, function(row) all(row == row[1], na.rm = TRUE))
    #if (any(identical_rows)) {
    #  for (i in which(identical_rows)) {
     #   expression_filtered[i, ] <- NaN
    #  }
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
  
  # Identify datasets for shared and secondary shared diseases
  target_diseases1 <- c(target_diseases, shared_diseases)
  #print(target_diseases1)
  #target_diseases1 <- c("psoriasis", "healthy")
  valid_shared_datasets <- subset(phenotype_data, disease %in% target_diseases1) %>%
    group_by(source_dataset) %>%
    summarize(num_categories = n_distinct(disease)) %>%
    filter(num_categories >= 2) %>%
    pull(source_dataset)
  
  disease_other <- valid_shared_datasets[valid_shared_datasets %in% valid_datasets]
  
  target_diseases2 <- c(shared_diseases,secondary_shared_diseases)
  #print(target_diseases2)
  #target_diseases2 <- c("healthy", "scleroderma")
  valid_secondary_shared_datasets <- subset(phenotype_data, disease %in% target_diseases2) %>%
    group_by(source_dataset) %>%
    summarize(num_categories = n_distinct(disease)) %>%
    filter(num_categories >= 2) %>%
    pull(source_dataset)
  disease_psoriasis <- valid_secondary_shared_datasets[valid_secondary_shared_datasets %in% valid_datasets]
  
  # Calculate similarity between datasets shared_diseases = "healthy"
  healthy_samples <- subset_data[subset_data$disease == shared_diseases, ]
  healthy_expression <- subset_expression[, healthy_samples$geo_accession]
  healthy_expression <- na.omit(healthy_expression)
  
  healthy_expression_by_dataset <- lapply(unique(healthy_samples$source_dataset), function(dataset_id) {
    dataset_healthy_samples <- healthy_samples[healthy_samples$source_dataset == dataset_id, ]
    healthy_expression[, dataset_healthy_samples$geo_accession, drop = FALSE]
  })
  
  names(healthy_expression_by_dataset) <- unique(healthy_samples$source_dataset)
  
  dataset_ids <- names(healthy_expression_by_dataset)
  n <- length(dataset_ids)
  dataset_similarity_matrix <- matrix(NA, nrow = n, ncol = n, dimnames = list(dataset_ids, dataset_ids))
  
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      dataset1 <- healthy_expression_by_dataset[[dataset_ids[i]]]
      dataset2 <- healthy_expression_by_dataset[[dataset_ids[j]]]
      
      if (is.null(dataset1) || is.null(dataset2) || ncol(dataset1) == 0 || ncol(dataset2) == 0) {
        message(paste("Invalid dataset pair:", dataset_ids[i], dataset_ids[j]))
        next
      }
      
      mean_dataset1 <- rowMeans(dataset1, na.rm = TRUE)
      mean_dataset2 <- rowMeans(dataset2, na.rm = TRUE)
      
      if (all(is.na(mean_dataset1)) || all(is.na(mean_dataset2))) {
        message(paste("Mean values for one of the datasets are NA. Skipping pair:", dataset_ids[i], dataset_ids[j]))
        next
      }
      
      similarity <- cor(mean_dataset1, mean_dataset2, method = "pearson", use = "complete.obs")
      
      dataset_similarity_matrix[i, j] <- similarity
      dataset_similarity_matrix[j, i] <- similarity
    }
  }
  
  
  # Find best matches between disease_other and disease_psoriasis datasets
  best_matches <- data.frame(
    disease_other = character(),
    best_disease_psoriasis = character(),
    similarity = numeric(),
    rank = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (other_dataset in disease_other) {
    similarities <- dataset_similarity_matrix[as.character(other_dataset), as.character(disease_psoriasis), drop = FALSE]
    
    sorted_indices <- order(similarities, decreasing = TRUE)
    top_indices <- sorted_indices[1:min(3, length(sorted_indices))]
    top_psoriasis_datasets <- colnames(similarities)[top_indices]
    top_similarities <- similarities[top_indices]
    
    for (rank in seq_along(top_psoriasis_datasets)) {
      best_matches <- rbind(best_matches, data.frame(
        disease_other = other_dataset,
        best_disease_psoriasis = top_psoriasis_datasets[rank],
        similarity = top_similarities[rank],
        rank = rank
      ))
    }
  }
  
  filtered_matches <- best_matches[best_matches$similarity > 0.6, ]
  
  if (nrow(filtered_matches) == 0) {
    filtered_matches <- best_matches[order(-best_matches$similarity), ][1:min(10, nrow(best_matches)), ]
  }
  
  filtered_matches <- na.omit(filtered_matches)
  # Perform differential analysis for matched datasets
  analysis_results <- list()
  #target_diseases = "psoriasis"
  ##secondary_shared_diseases = "scleroderma"
  for (i in seq_len(nrow(filtered_matches))) {
    lupus_dataset <- as.character(filtered_matches$disease_other[i])
    psoriasis_dataset <- as.character(filtered_matches$best_disease_psoriasis[i])

    selected_samples <- subset_data$source_dataset %in% c(lupus_dataset, psoriasis_dataset)
    current_data <- subset_data[selected_samples, ]
    current_expression <- subset_expression[, current_data$geo_accession]
    current_expression <- na.omit(current_expression)
      

    #print(levels(current_data$disease))
    current_data$disease <- factor(current_data$disease)
    design <- model.matrix(~ 0 + disease, data = current_data)
    colnames(design) <- make.names(levels(current_data$disease))
    

    #print(paste("Design matrix columns:", colnames(design)))
    
    batch <- factor(current_data$source_dataset)
    mod <- model.matrix(~ disease, data = current_data)
    
    batch_corrected_expression <- removeBatchEffect(
      current_expression, 
      batch = batch, 
      design = mod
    )
    

    #batch_corrected_expression <- batch_corrected_expression[rowVars(as.matrix(batch_corrected_expression)) > 1e-6, ]
    

    target_diseases3 <- unique(c(target_diseases, secondary_shared_diseases))
    current_data <- current_data[current_data$disease %in% target_diseases3, ]
    batch_corrected_expression <- batch_corrected_expression[, current_data$geo_accession]
    current_data$disease <- factor(current_data$disease, levels = c(target_diseases, secondary_shared_diseases))
    

    
    #unique(current_data$disease)
    design <- model.matrix(~ 0 + disease, data = current_data)
    colnames(design) <- make.names(c(target_diseases, secondary_shared_diseases)) #make.names(levels(current_data$disease))
    rownames(design) <- current_data$geo_accession
    #print(design)
    

    contrast_name <- paste0(colnames(design)[1], " - ", colnames(design)[2])    #print(paste("Contrast name:", contrast_name))
    #print(paste("Contrast name:", contrast_name))
    contrast <- makeContrasts(contrasts = contrast_name, levels = design)

    fit <- lmFit(batch_corrected_expression, design)
    fit <- eBayes(fit)

    fit2 <- contrasts.fit(fit, contrast)
    fit2 <- eBayes(fit2)
    

    result <- topTable(fit2, adjust = "fdr", number = Inf)
    #result$ID[duplicated(result$ID)]
    #rownames(result) <- result$ID
    

    analysis_results[[paste(lupus_dataset, psoriasis_dataset, sep = "_vs_")]] <- result
    
  }
  
  all_genes <- rownames(subset_expression) #unique(unlist(lapply(analysis_results, rownames)))
  

  merged_logFC <- as.data.frame(matrix(NA, nrow = length(all_genes), ncol = length(analysis_results)))
  rownames(merged_logFC) <- all_genes
  colnames(merged_logFC) <- names(analysis_results)
  

  for (result_name in names(analysis_results)) {
    result <- analysis_results[[result_name]]
    merged_logFC[rownames(result), result_name] <- result$logFC
  }
  

  average_logFC <- rowMeans(merged_logFC, na.rm = TRUE)
  

  average_logFC_df <- data.frame(
    Gene = rownames(merged_logFC),
    Average_logFC = average_logFC
  )
  

  median_value <- median(average_logFC_df$Average_logFC, na.rm = TRUE)
  
 
  average_logFC_df$Average_logFC[is.na(average_logFC_df$Average_logFC)] <- median_value
  

  #secondary_shared_results[[secondary_shared_diseases]]  <- average_logFC_df
  return(average_logFC_df)
  
}


analyze_logFC_with_shared_categories <- function(phenotype_data, expression_data, target_disease, similarity_threshold = 0.3) {

  shared_categories <- phenotype_data %>%
    filter(disease != target_disease) %>%
    group_by(disease) %>%
    filter(any(source_dataset %in% phenotype_data$source_dataset[phenotype_data$disease == target_disease])) %>%
    pull(disease) %>%
    unique()
  

  shared_results <- list()
  secondary_shared_results <- list()
  no_shared_results <- list()
  

  secondary_shared_categories <- phenotype_data %>%
    filter(!(disease %in% c(target_disease, shared_categories))) %>%
    group_by(disease) %>%
    filter(any(source_dataset %in% phenotype_data$source_dataset[phenotype_data$disease %in% shared_categories])) %>%
    pull(disease) %>%
    unique()
  

  all_diseases <- unique(phenotype_data$disease)
  no_shared_categories <- setdiff(all_diseases, c(target_disease, shared_categories, secondary_shared_categories))

  for (shared_disease in shared_categories) {
    print(paste("当前是一级共享类：", shared_disease))
    

    valid_datasets <- phenotype_data %>%
      filter(disease %in% c(target_disease, shared_disease)) %>%
      group_by(source_dataset) %>%
      summarize(num_categories = n_distinct(disease)) %>%
      filter(num_categories == 2) %>%
      pull(source_dataset)
    

    diff_analysis_results <- list()
    
    for (dataset_id in valid_datasets) {

      selected_samples <- phenotype_data$source_dataset == dataset_id & 
        phenotype_data$disease %in% c(target_disease, shared_disease)
      phenotype_filtered <- phenotype_data[selected_samples, ]
      expression_filtered <- expression_data[, selected_samples]
      
      # 检查是否包含所有类别
      if (!all(c(target_disease, shared_disease) %in% unique(phenotype_filtered$disease))) {
        message(paste("Dataset", dataset_id, "does not contain both target and shared categories. Skipping."))
        next
      }
      

      identical_rows <- apply(expression_filtered, 1, function(row) all(row == row[1], na.rm = TRUE))
      
      if (any(identical_rows)) {
        #message(paste("Dataset", dataset_id, "contains rows with identical values. Replacing them with median values."))
  
       # median_values <- apply(expression_filtered[identical_rows, , drop = FALSE], 1, median, na.rm = TRUE)
        for (i in which(identical_rows)) {
          expression_filtered[i, ] <- 0
        }
      }
      

      qx <- as.numeric(quantile(expression_filtered, c(0., 0.25, 0.5, 0.75, 0.95, 1.0), na.rm = TRUE))
      LogC <- (qx[5] > 100) || (qx[6] - qx[1] > 50 && qx[2] > 0)
      if (LogC) {

        #if (any(expression_filtered < 0, na.rm = TRUE)) {
       #   min_value <- abs(min(expression_filtered, na.rm = TRUE))
       #   expression_filtered <- expression_filtered + min_value + 1
       # }        
        expression_filtered[expression_filtered <= 0] <- NaN
        expression_filtered <- log2(expression_filtered)

        expression_filtered[is.nan(as.matrix(expression_filtered))] <- 0        
      }
      
      # Replace NA values with row means
      #expression_filtered <- as.data.frame(apply(expression_filtered, 1, function(row) {
      #  row_mean <- mean(row, na.rm = TRUE)
      #  row[is.na(row)] <- row_mean
      #  return(row)
      #}))
      expression_filtered <- as.data.frame(expression_filtered)
      

      sample_counts <- table(phenotype_filtered$disease)
      
      if (all(sample_counts == 1)) {

        target_sample <- expression_filtered[, phenotype_filtered$disease == target_disease]
        comparison_sample <- expression_filtered[, phenotype_filtered$disease == shared_disease]
        logFC <- log2(target_sample + 1) - log2(comparison_sample + 1)


        result <- data.frame(
          Gene = rownames(expression_filtered),
          logFC = logFC
        )
        rownames(result) <- rownames(expression_filtered)
        diff_analysis_results[[as.character(dataset_id)]] <- result
        next
      }
      

      phenotype_filtered$disease <- factor(phenotype_filtered$disease, levels = c(target_disease, shared_disease))
      

      design <- model.matrix(~ 0 + disease, data = phenotype_filtered)
      colnames(design) <- make.names(c(target_disease, shared_disease))
      contrast <- makeContrasts(
        contrasts = paste0("`", make.names(target_disease), "` - `", make.names(shared_disease), "`"),
        levels = design
      )
      
      fit <- lmFit(expression_filtered, design)
      fit <- contrasts.fit(fit, contrast)
      fit <- eBayes(fit)
      diff_results <- topTable(fit, adjust = "fdr", number = Inf)
      diff_analysis_results[[as.character(dataset_id)]] <- diff_results
    }
    

    all_genes <- unique(unlist(lapply(diff_analysis_results, rownames)))
    merged_logFC <- as.data.frame(matrix(NA, nrow = length(all_genes), ncol = length(diff_analysis_results)))
    rownames(merged_logFC) <- all_genes
    colnames(merged_logFC) <- names(diff_analysis_results)
    
    for (dataset_id in names(diff_analysis_results)) {
      logFC_data <- diff_analysis_results[[dataset_id]]
      merged_logFC[rownames(logFC_data), dataset_id] <- logFC_data$logFC
    }


    if (ncol(merged_logFC) == 1) {

      average_logFC <- as.data.frame(merged_logFC)
      colnames(average_logFC) <- shared_disease
    } else if  (ncol(merged_logFC) <= 3) {

      average_logFC <- as.data.frame(rowMeans(merged_logFC, na.rm = TRUE))
      colnames(average_logFC) <- shared_disease
    } else {

      similarity_matrix <- cor(merged_logFC, use = "pairwise.complete.obs")
      

      datasets_to_keep <- colnames(similarity_matrix)[apply(similarity_matrix, 1, function(row) {
        any(row >= similarity_threshold & row < 1, na.rm = TRUE)
      })]
      
    
      if (length(datasets_to_keep) == 0) {
        datasets_to_keep <- colnames(merged_logFC)
        message("No datasets meet the similarity threshold. Using all datasets.")
      }
      
 
      filtered_logFC_matrix <- merged_logFC[, datasets_to_keep, drop = FALSE]
      

      average_logFC <- as.data.frame(rowMeans(filtered_logFC_matrix, na.rm = TRUE))
      colnames(average_logFC) <- shared_disease
      
    }
    
    average_logFC_df <- data.frame(
      Gene = rownames(average_logFC),
      Average_logFC = average_logFC )  
      


    shared_results[[shared_disease]] <- average_logFC_df
  }
  

  for (secondary_shared_disease  in secondary_shared_categories) {
    print(paste("当前是二级共享类：", secondary_shared_disease))
    associated_datasets <- phenotype_data %>%
      filter(disease == secondary_shared_disease) %>%
      pull(source_dataset) %>%
      unique()
    
    

    shared_diseases_for_secondary <- phenotype_data %>%
      filter(source_dataset %in% associated_datasets & disease %in% shared_categories) %>%
      group_by(disease) %>%
      summarize(num_samples = n()) %>%
      arrange(desc(num_samples)) %>%
      pull(disease)
    
    if (length(shared_diseases_for_secondary) == 0) {
      message(paste("No valid shared categories for secondary shared disease:", secondary_shared_disease))
      next
    }
    
    most_shared_disease <- shared_diseases_for_secondary[1]
    
    result <- process_differential_analysis(
      target_diseases = target_disease,
      shared_diseases = most_shared_disease,
      secondary_shared_diseases = secondary_shared_disease,
      phenotype_data = phenotype_data,
      expression_data = expression_data
    )

    secondary_shared_results[[secondary_shared_disease]] <- result

    
  }
  
  for (no_shared_disease in no_shared_categories) {
    message(paste("Processing target disease:", target_disease, "with no shared category:", no_shared_disease))
    #no_shared_disease = "actinic lentigines.nonlesional"
    #no_shared_disease = "actinic lentigines"
    valid_datasets <- phenotype_data %>%
      filter(disease %in% c(target_disease, no_shared_disease)) %>%
      group_by(source_dataset) %>%
      summarize(num_categories = n_distinct(disease)) %>%
      filter(num_categories == 1) %>%
      pull(source_dataset)
    
    #dataset_id  = 19
    for (dataset_id in valid_datasets) {
      #message(paste("Processing dataset:", dataset_id))
      
      phenotype_data$disease <- as.character(phenotype_data$disease)
      selected_samples <- phenotype_data$source_dataset == dataset_id & 
        phenotype_data$disease %in% c(as.character(target_disease), as.character(no_shared_disease))
      #selected_samples <- phenotype_data$source_dataset == dataset_id & phenotype_data$disease %in% c(target_disease, no_shared_disease)
      expression_filtered <- expression_data[, selected_samples]
      #sum(selected_samples)
      #names(expression_filtered)
      qx <- as.numeric(quantile(expression_filtered, c(0., 0.25, 0.5, 0.75, 0.95, 1.0), na.rm = TRUE))
      LogC <- (qx[5] > 100) || (qx[6] - qx[1] > 50 && qx[2] > 0)
      

      if (LogC) {
        expression_filtered[expression_filtered <= 0] <- NaN
        expression_filtered <- log2(expression_filtered)
        expression_filtered[is.nan(as.matrix(expression_filtered))] <- 0
        
      }
      
      expression_filtered <- as.data.frame(expression_filtered)
      colnames(expression_filtered) <- phenotype_data[selected_samples,]$geo_accession
      rownames(expression_filtered) <- rownames(expression_data)
        
      if (!exists("dataset_transformed_expression")) {
        dataset_transformed_expression <- expression_filtered
      } else {
        dataset_transformed_expression <- cbind(dataset_transformed_expression, expression_filtered)
      }
    }
    
    expression_filtered <- dataset_transformed_expression
    rm(dataset_transformed_expression)
    

    phenotype_filtered <- phenotype_data[phenotype_data$geo_accession %in% colnames(expression_filtered), ]
    

    expression_filtered <- expression_filtered[, phenotype_filtered$geo_accession, drop = FALSE]
   
    
    #sum(phenotype_filtered$geo_accession == colnames(expression_filtered))
    

   # combat_expression <- expression_filtered
    #cat("Rows in phenotype_filtered:", nrow(phenotype_filtered), "\n")
    #cat("Columns in expression_filtered:", ncol(expression_filtered), "\n")
    #phenotype_filtered$disease <- droplevels(phenotype_filtered$disease)
   # cat("Levels in disease:", levels(phenotype_filtered$disease), "\n")
    #na_samples <- phenotype_filtered[is.na(phenotype_filtered$disease), ]
    #cat("Samples with NA disease values:\n")
    #print(na_samples)
    
    

    phenotype_filtered$disease <- factor(phenotype_filtered$disease, levels = c(target_disease, no_shared_disease))
    #unique(phenotype_filtered$disease)

    design <- model.matrix(~ 0 + disease, data = phenotype_filtered)
    colnames(design) <- make.names(c(target_disease, no_shared_disease))
    rownames(design) <- phenotype_filtered$geo_accession
    #nrow(design) length(phenotype_filtered$geo_accession)

    contrast <- makeContrasts(
      contrasts = paste0("`", make.names(target_disease), "` - `", make.names(no_shared_disease), "`"),
      levels = design
    )
    fit <- lmFit(expression_filtered, design)
    fit <- contrasts.fit(fit, contrast)
    fit <- eBayes(fit)
    diff_results <- topTable(fit, adjust = "fdr", number = Inf)

    logFC_data <- data.frame(
      Gene = rownames(diff_results),
      logFC = diff_results$logFC
    )
    
    no_shared_results[[no_shared_disease]] <- logFC_data
    #print(paste("当前是无共享类：", no_shared_disease))    
  }
  
  

  list(
    shared_results = shared_results,
    secondary_shared_results = secondary_shared_results,
    no_shared_results = no_shared_results
  )
}


run_shared_category_logFC_analysis <- function(
    phenotype_data,
    expression_data,
    target_disease,
    output_path,
    similarity_threshold = 0.3
) {
  library(dplyr)
  library(limma)
  

  result <- analyze_logFC_with_shared_categories(
    phenotype_data = phenotype_data,
    expression_data = expression_data,
    target_disease = target_disease,
    similarity_threshold = similarity_threshold
  )
  

  merge_results <- function(results_list) {
    if (length(results_list) == 0) {
      return(data.frame(Gene = character()))  
    }
    merged_df <- NULL
    for (name in names(results_list)) {
      df <- results_list[[name]]
      colnames(df)[colnames(df) == "Average_logFC"] <- name
      colnames(df)[colnames(df) == "logFC"] <- name
      if (is.null(merged_df)) {
        merged_df <- df
      } else {
        merged_df <- merge(merged_df, df, by = "Gene", all = TRUE)
      }
    }
    return(merged_df)
  }
  

  merged_shared_results <- merge_results(result$shared_results)
  merged_secondary_shared_results <- merge_results(result$secondary_shared_results)
  merged_no_shared_results <- merge_results(result$no_shared_results)
  


  merged_list <- list(merged_shared_results, merged_secondary_shared_results, merged_no_shared_results)
  merged_list <- Filter(function(df) is.data.frame(df) && ncol(df) > 1, merged_list)


  if (length(merged_list) == 0) {
    warning("⚠️ 所有差异分析结果都是空的，无法合并！")
    final_merged_results <- data.frame()
  } else {
    final_merged_results <- Reduce(function(x, y) merge(x, y, by = "Gene", all = TRUE), merged_list)
  }

  

  rownames(final_merged_results) <- final_merged_results$Gene
  final_merged_results <- final_merged_results[, -1]
  

  final_merged_results <- final_merged_results * -1
  

  if (exists("phenotype_data1")) {
    disease_unique <- unique(phenotype_data1$disease)
    clean_column_names <- gsub("[^[:alpha:]]", "", colnames(final_merged_results))
    clean_disease_names <- gsub("[^[:alpha:]]", "", disease_unique)
    new_column_names <- colnames(final_merged_results)
    
    for (i in seq_along(clean_column_names)) {
      match_index <- which(clean_column_names[i] == clean_disease_names)
      if (length(match_index) > 0) {
        new_column_names[i] <- disease_unique[match_index[1]]
      }
    }
    colnames(final_merged_results) <- new_column_names
  }
  

  output_file <- file.path(output_path, paste0(gsub("[^a-zA-Z0-9]", "_", target_disease), "_merged_results.csv"))
  write.csv(final_merged_results, file = output_file, row.names = TRUE)
  

  return(list(
    merged_results = final_merged_results,
    result = result,
    output_file = output_file
  ))
}
