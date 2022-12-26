library(corrplot)

# the number of samples
sample_size <- 500
# the number of predictors
predictor_size <- 500
group_size <- 5
# the number of predictors truly associated with a response variable
significant_predictors <- floor(predictor_size*sample((50:200)/1000,1)) 

## set the predictors associated with an outcome
### the number of predictors positively associated with an outcome
### the number of predictors negatively associated with an outcome
positively_associated_predictors<-floor(significant_predictors*0.4) 
negatively_associated_predictors<-significant_predictors-positively_associated_predictors 

## set correlated predictors within each group
### randomly sampling proportions of 10 correlated predictor groups 
### to become their sum equal to 1
proportion_list<-sample(seq(1,1+2*(100-group_size)/group_size,
                            by=2*(100-group_size)/(group_size*(group_size-1)))/100,
                        group_size,replace=FALSE)%>%round(3) 
names(proportion_list)<-paste0("group",1:length(proportion_list))
### initialize a matrix with a size as sample_size by predictor_size
predictor_matrix <- matrix(0, ncol = predictor_size, nrow = sample_size)
### initialize a data frame and assign meta information used to generate simulated data
meta_data<-
    data.frame(group_name=c(names(proportion_list)),
               proportion=proportion_list)%>%
    mutate(group_n=(predictor_size*proportion_list)%>%round(0), # the within-group number of predictors
           first_index=c(1,cumsum(group_n[-length(proportion_list)])+1), # the 1st index of predictors in each group
           last_index=cumsum(group_n), # the last index of predictors in each group
           group_correlation=sample((0:700)/1000,length(proportion_list),replace=TRUE), # correlation among the within-group predictors
           group_effect=sample((-500:500)/1000,length(proportion_list),replace=TRUE)); # effect of each group on an outcome variable


data<-matrix(rnorm(sample_size*predictor_size,mean=0,sd=0.01), 
             nrow = sample_size, ncol = predictor_size)
covariance_matrix<-matrix(rnorm(predictor_size*predictor_size,0.15,0.05),
                          nrow=predictor_size, ncol=predictor_size)
beta_coefficients <- rnorm(predictor_size,0,0.1)%>%vector()


for (i in 1:nrow(meta_data)) {
    
    group_range <- meta_data[i, "first_index"]:meta_data[i, "last_index"]
    for (j in group_range){
        for(k in group_range){
        covariance_matrix[j, k] <- meta_data[i, "group_correlation"]
        }
    }
    #covariance_matrix[group_range, group_range]+meta_data[i, "group_correlation"]    
    diag(covariance_matrix) <- 1
    data[, group_range] <- 
        mvrnorm(n = sample_size, 
                mu = rep(0,meta_data[i,"group_n"]),
                Sigma = covariance_matrix[group_range, group_range])
    data=as.data.frame(data)
    beta_coefficients[meta_data[i, "first_index"]:meta_data[i, "last_index"]] <-
        beta_coefficients[meta_data[i, "first_index"]:meta_data[i, "last_index"]]+
        meta_data[i,"group_effect"]
    predictor_names<-paste0(meta_data[i,"group_name"],"_",1:meta_data[i,"group_n"])
    names(beta_coefficients[meta_data[i, "first_index"]:meta_data[i, "last_index"]]) <- predictor_names
    colnames(data[meta_data[i, "first_index"]:meta_data[i, "last_index"]])<-predictor_names
        
}

score=data%*%beta_coefficients # score of each sample
p <- 1/(1+exp(-(0+score)))  # logistic function to get a probability, intercept = 0
outcome <- rbinom(sample_size, 1, p)

rbinom(sample_size, 1, p)%>%table


