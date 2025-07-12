from .deepadvancer import (
    load_and_process_expression_data, #整合数据
    run_logfc_analysis_and_generate_fc_array, #重构数据
    compute_logfc_between_classes, #比较类别之间
    compute_logfc_vs_others #比较一个类与所有类
)