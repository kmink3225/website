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
    data.frame(group_name=c(names(proportion_list)))%>%
    mutate(group_n=(predictor_size*proportion_list)%>%round(0), # the within-group number of predictors
           first_index=c(1,cumsum(group_n[-length(proportion_list)])+1), # the 1st index of predictors in each group
           last_index=cumsum(predictor_size*proportion_list), # the last index of predictors in each group
           group_correlation=sample((0:700)/1000,length(proportion_list),replace=TRUE), # correlation among the within-group predictors
           group_effect=sample((-500:500)/1000,length(proportion_list),replace=TRUE)); # effect of each group on an outcome variable


data<-matrix(rnorm(sample_size*predictor_size,mean=0,sd=0.01), nrow = sample_size, ncol = predictor_size)
covariance_matrix<-matrix(rnorm(predictor_size*predictor_size,0.15,0.05),
                          nrow=predictor_size,ncol=predictor_size)
covariance_matrix<-matrix(0,
                          nrow=predictor_size,ncol=predictor_size)
beta_coefficients <- rep(0, predictor_size)

i=2
for (i in 1:nrow(meta_data)) {
    
    group_range <- meta_data[i, "first_index"]:meta_data[i, "last_index"]
    for (j in group_range){
        for(k in group_range){
        covariance_matrix[j, k] <- meta_data[i, "group_correlation"]
        print(covariance_matrix[j, k])
        }
    }
    print(i)
    #covariance_matrix[group_range, group_range]+meta_data[i, "group_correlation"]    
    diag(covariance_matrix) <- 1
    data[, group_range] <- 
        mvrnorm(n = sample_size, 
                mu = rep(0,predictor_size*proportion_list[i]),
                Sigma = covariance_matrix[group_range, group_range])
    
    beta_coefficients[meta_data[i, "first_index"]:meta_data[i, "last_index"]] <- meta_data[i,"group_effect"]
}
View(covariance_matrix)
corrplot(covariance_matrix)
corrplot(cor(covariance_matrix[group_range, group_range]))









nvars = c(10, 10, 10, 10, 60)
cors = c(0.1, 0.2, 0.3, 0.4, 0.5)
associations = c(0.5, 0.5, 0.3, 0.3, 0)
firstonly = c(TRUE,FALSE, TRUE, FALSE, FALSE)
nsamples = 100
censoring = "none"
labelswapprob = 0
response = "timetoevent"
basehaz = 0.2
logisticintercept = 0
    
x.out <- matrix(0, ncol = sum(nvars), nrow = nsamples)
definecors <- data.frame(start = c(1, cumsum(nvars[-length(nvars)]) +
                                       1), end = cumsum(nvars), cors = cors, associations = associations,
                         num = nvars, firstonly = firstonly, row.names = letters[1:length(nvars)])
Sigma <- matrix(0, ncol = sum(nvars), nrow = sum(nvars))
wts <- rep(0, sum(nvars))
for (i in 1:nrow(definecors)) {
    thisrange <- definecors[i, "start"]:definecors[i, "end"]
    Sigma[thisrange, thisrange] <- definecors[i, "cors"]
    diag(Sigma) <- 1
    x.out[, thisrange] <- mvrnorm(n = nsamples, mu = rep(0,
                                                         nvars[i]), Sigma = Sigma[thisrange, thisrange])
    if (definecors[i, "firstonly"]) {
        wts[definecors[i, "start"]] <- definecors[i, "associations"]
    }
    else {
        wts[definecors[i, "start"]:definecors[i, "end"]] <- definecors[i,
                                                                       "associations"]
    }
    varnames <- paste(letters[i], 1:nvars[i], sep = ".")
    names(wts)[definecors[i, "start"]:definecors[i, "end"]] <- varnames
}
View(Sigma)
corrplot(Sigma)
names(wts) <- make.unique(names(wts))
dimnames(Sigma) <- list(colnames = names(wts), rownames = names(wts))
colnames(x.out) <- names(wts)
betaX <- x.out %*% wts
x.out <- data.frame(x.out)
if (identical(response, "timetoevent")) {
    h = basehaz * exp(betaX[, 1])
    x.out$time <- rexp(length(h), h)
    x.out$cens <- 1
    if(is(censoring, "numeric")){
        if(length(censoring)==2){
            censtimes <- runif(length(h),min=censoring[1],max=censoring[2])
        }else if(length(censoring)==1){
            censtimes <- rep(censoring,length(h))
        }
        x.out$cens[x.out$time>censtimes] <- 0
        x.out$time[x.out$time>censtimes] <- censtimes[x.out$time>censtimes]
    }
}
else if (identical(response, "binary")) {
    p <- 1/(1 + exp(-(betaX + logisticintercept)))
    x.out$outcome <- rbinom(length(p), 1, p)
    if(labelswapprob > 0){
        do.swap <- runif(length(p)) < labelswapprob
        new.outcome <- x.out$outcome
        new.outcome[x.out$outcome==1 & do.swap] <- 0
        new.outcome[x.out$outcome==0 & do.swap] <- 1
        x.out$outcome <- new.outcome
    }
    x.out$outcome <- factor(x.out$outcome)
}
else stop("response must be either timetoevent or binary")
return(list(summary = definecors, associations = wts, covariance = Sigma,
            data = x.out))
}

View(Sigma)
