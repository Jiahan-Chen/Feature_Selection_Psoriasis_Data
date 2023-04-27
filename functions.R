


# wrapper function for feature selection
# x: the input data frame, n rows (samples), p columns (features)
# method: method of feature selection
# m: number of selected features
#
FS_wrapper <- function(x, method, m = NULL){
  if(method == "PCA"){
    if (!is.null(m)){
      return(prcomp(x, scale. = TRUE, center = TRUE)$rotation[,1:m])
    } else{
      return(prcomp(x, scale. = TRUE, center = TRUE)$rotation)
    }
    
  } else if(method == "UMAP"){
    require(uwot)
    if (!is.null(m)){
      return(uwot::umap(t(x),n_components = m))
    } else{
      return(uwot::umap(t(x)))
    }
  }
  
}

# x: counts of microarray or RNA-seq data, n x p, n: number of samples, p: number of features
cluster_wrapper <- function(x, method, K_search = c(2:8), K = NULL){
  if(K_search[1]==1){K_search=K_search[-1]}  # remove K = 1
  
  if (method == 'hc'){
    require(NbClust)
    # dissimilar matrix
    d = as.dist(1-cor(t(x), method="spearman"))
    if(is.null(K)){
      results_HC = NbClust(data = x, diss = d, distance = "NULL", min.nc = min(K_search), max.nc = max(K_search), method = 'average', index = 'gap')
      K = results_HC$Best.nc[1]
    } 
    cls = cutree(hclust(d,method="average"),K)
    return(list(
      fit = results_HC,
      K = K,
      cls = cls
    ))
  } else if (method =='km'){
    require(cluster)
    if(is.null(K)){
      pam_fit = list()
      sil.vals = rep(NA,length(K_search))
      
      for(c in 1:length(K_search)){
        pam_fit[[c]] = pam(x,K_search[c])
        sil.vals[c] = pam_fit[[c]]$silinfo$avg.width
        cat(paste("KM order select avg silhouette: ", sil.vals[c],"\n"))
      }
      K = K_search[which.max(sil.vals)]
      fit = pam_fit[[which.max(sil.vals)]]
    } else{
      fit = pam(x,K)
    }
    cls = fit$cluster
    return(list(
      fit = fit,
      K = K,
      cls = cls
    ))
  } else if (method == "mc"){
    require(mclust)
    if(is.null(K)){
      fit=Mclust(x,G=K_search)
    } else {
      fit=Mclust(x,G=K)
    }
    return(list(
      fit=fit,
      K=fit$G,
      cls=fit$classification
    ))
  }
}
  
simulation_1 <- function(expression, annotation, size_1, size_2){
  index_1 = which(annotation == "involved skin from cases")
  index_2 = which(annotation == "normal skin from controls")
  sample_1 = sample(index_1, size = size_1, replace = FALSE)
  sample_2 = sample(index_2, size = size_2, replace = FALSE)
  sim_expr = expression[,c(sample_1, sample_2)]
  sim_pheno = annotation[c(sample_1, sample_2)]
  return(list(sim_expr, sim_pheno))
}

jaccard_index <- function(prediction, label){
  # rename label
  label_rename1 = ifelse(label == "involved skin from cases", 1, 2)
  label_rename2 = ifelse(label == "involved skin from cases", 2, 1)
  
  intersection1 = sum(prediction == label_rename1)
  intersection2 = sum(prediction == label_rename2)
  
  union1 = length(prediction) + length(label_rename1) - intersection1
  union2 = length(prediction) + length(label_rename2) - intersection2
  return (max(intersection1/union1, intersection2/union2))
}



# R 3.0.2
#library(LRAcluster)
