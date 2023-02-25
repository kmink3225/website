## Function List
color_function<-function(category_number){
return(
    if(category_number==2){
        c("darkblue","darkred")
    }else if(category_number==3){
        c("darkblue","darkred","yellow4")
    }else if(category_number==4){
        c("darkblue","darkred","yellow4","blueviolet")
    }else if(category_number==5){
        c("darkblue","darkred","yellow4","blueviolet","darkorange")
    }else{
        c("darkblue","darkred","yellow4","blueviolet","darkorange","darkgreen")
    }
    )
}

scale_function=function(vector=x,min=NULL,max=NULL,method){
    scaling_methods<-c('min_max normalization','customized normalization','standardization')

    if(method=="min-max"){
        output=(vector-min(vector))/(max(vector)-min(vector))
    }else if(method=="customized"){
        output=(max-min)*(vector-min(vector))/(max(vector)-min(vector))+min
    }else if(method=="standarized"){
        output=(vector-mean(vector))/sd(vector)
    }else{
        output=paste0("Error!, no such a scaling method in this module. \n Please, put the first word of each method you want to use in the 'method' argument among the following tests: ", paste(scaling_methods,collapse=", "))
    }
  return(output)
}

multiple_shapiro_test<-function(in_data){
        normality_test<-apply(in_data[,unlist(lapply(in_data, is.numeric))],2,
                            function(x)shapiro.test(x))
        temp<-data.frame(matrix(nrow=length(normality_test),ncol=4))
        for (i in 1:length(normality_test)){
            temp[i,]<-c(
                coloumn_name=names(normality_test)[i],
                statistic=normality_test[[i]]$statistic,
                p_value=normality_test[[i]]$p.value,
                method=normality_test[[i]]$method)
        }
        names(temp)<-c('column_name','statistic','p_value','method')
        output<-temp%>%
            mutate(p_adjusted=p.adjust(p_value,method="bonferroni"),
            type=ifelse(p_adjusted<0.05,'not_normal','normal'))%>%
            dplyr::select('column_name','statistic','p_value','p_adjusted','type','method')
        return(output)
}    

multiple_levene_test<-function(in_data,categorical_variable){
        homoscedasticity_test<-apply(in_data[,unlist(lapply(in_data, is.numeric))],2,
                                    function(x)leveneTest(x~in_data[,categorical_variable]))
        temp<-data.frame(matrix(nrow=length(homoscedasticity_test),ncol=6))
            for (i in 1:length(homoscedasticity_test)){
                temp[i,]<-c(
                    coloumn_name=names(homoscedasticity_test)[i],
                    group_df=homoscedasticity_test[[i]]$Df[1],
                    residual_df=homoscedasticity_test[[i]]$Df[2],
                    statistic=homoscedasticity_test[[i]]$`F value`[1],
                    p_value=homoscedasticity_test[[i]]$`Pr(>F)`[1],
                    method="levene's test")
            }
            names(temp)<-c('column_name','group_df','residual_df','statistic','p_value','method')
            output<-temp%>%
                mutate(p_adjusted=p.adjust(p_value,method="bonferroni"),
                type=ifelse(p_adjusted<0.05,'heteroscedasticity','homoscedasticity'))%>%
                dplyr::select('column_name','group_df','residual_df','statistic','p_value','p_adjusted','type','method')
        return(output)} 

multiple_unpaired_t_test<-function(in_data,categorical_variable,homo_variables,hetero_variables){
    homo_unpaired_t_test<-apply(in_data[,unlist(lapply(in_data, is.numeric))][,homo_variables],2,
                                    function(x)t.test(x~in_data[,categorical_variable],var.equal=TRUE))
    hetero_unpaired_t_test<-apply(in_data[,unlist(lapply(in_data, is.numeric))][,hetero_variables],2,
                                    function(x)t.test(x~in_data[,categorical_variable],var.equal=FALSE)) 
    unpaired_t_test<-c(homo_unpaired_t_test,hetero_unpaired_t_test)

    temp<-data.frame(matrix(nrow=length(unpaired_t_test),ncol=7))
        for (i in 1:length(unpaired_t_test)){
            temp[i,]<-c(names(unpaired_t_test)[i], 
                        unpaired_t_test[[i]]$estimate,
                        unpaired_t_test[[i]]$parameter,
                        unpaired_t_test[[i]]$statistic,
                        unpaired_t_test[[i]]$p.value,
                        unpaired_t_test[[i]]$method)
        }
        names(temp)<-c('column_name',names(unpaired_t_test[[1]]$estimate),'df','statistic','p_value','method')
        output<-temp%>%
            mutate(p_adjusted=p.adjust(p_value,method="bonferroni"),
            type=ifelse(p_adjusted<0.05,'significant','insignificant'))%>%
            dplyr::select('column_name',names(unpaired_t_test[[1]]$estimate),'df','statistic','p_value','p_adjusted','type','method')
    return(output)} 


multiple_correlation_test<-function(in_data,in_numeric_variable){
    correlation_test<-apply(in_data[,unlist(lapply(in_data, is.numeric))],2,
                                    function(x)cor.test(x,in_data[,in_numeric_variable],method='pearson'))
    temp<-data.frame(matrix(nrow=length(correlation_test),ncol=6))
        for (i in 1:length(correlation_test)){
            temp[i,]<-c(names(correlation_test)[i], 
                        correlation_test[[i]]$estimate,
                        correlation_test[[i]]$parameter,
                        correlation_test[[i]]$statistic,
                        correlation_test[[i]]$p.value,
                        correlation_test[[i]]$method)
        }
        names(temp)<-c('column_name',names(correlation_test[[1]]$estimate),'df','statistic','p_value','method')
        output<-temp%>%
            mutate(p_adjusted=p.adjust(p_value,method="bonferroni"),
            type=ifelse(p_adjusted<0.05,'significant','insignificant'))%>%
            dplyr::select('column_name',names(correlation_test[[1]]$estimate),'df','statistic','p_value','p_adjusted','type','method')
    return(output)} 

multiple_anova_test<-function(in_data, in_categorical_variable){
    aov_test<-apply(in_data[,unlist(lapply(in_data, is.numeric))],2,
                function(x)aov(x~get(in_categorical_variable),data=in_data)%>%summary)

    temp<-data.frame(matrix(nrow=length(aov_test),ncol=10))
    for (i in 1:length(aov_test)){
        temp[i,]<-c(names(aov_test)[i], 
                    aov_test[[i]][[1]]$`Df`[1],
                    aov_test[[i]][[1]]$`Df`[2],
                    aov_test[[i]][[1]]$`Sum Sq`[1],
                    aov_test[[i]][[1]]$`Sum Sq`[2],
                    aov_test[[i]][[1]]$`Mean Sq`[1],
                    aov_test[[i]][[1]]$`Mean Sq`[2],
                    aov_test[[i]][[1]]$`F value`[1],
                    aov_test[[i]][[1]]$`Pr(>F)`[1],
                    'one_way_anova')
    }
    names(temp)<-c('column_name','group_df','residual_df','group_ssq','residual_ssq',
                    'group_msq','residual_msq','F_value','p_value','method')
    output<-temp%>%
            mutate(p_adjusted=p.adjust(p_value,method="bonferroni"),
            type=ifelse(p_adjusted<0.05,'significant','insignificant'))%>%
            dplyr::select('column_name','group_df','residual_df','group_ssq','residual_ssq',
                    'group_msq','residual_msq','F_value','p_value','method',
                    'p_adjusted','type','method')
    return(output)} 

main_statistical_test<-function(
    in_data,method,categorical_variable,in_numeric_variable,
    homo_variables=NULL,hetero_variables=NULL,
    fun1=multiple_shapiro_test,
    fun2=multiple_levene_test,
    fun3=multiple_unpaired_t_test){
    test_list<-c("shapiro wilks test","levene's test","student t test","anova","correlation test")#,"ANCOVA","MANOVA","wilcoxon manwhitney","kruskal wallis test","fisher exact test","anderson darling")
    error_massage<-paste0("Error!, no such a test in this module. \n Please, put the first word of each method you want to use in the 'method' argument among the following tests: ", paste(test_list,collapse=", "))
    if(grepl('shapiro',method)){
        output=multiple_shapiro_test(in_data)
    }else if(grepl('levene',method)){
        output=multiple_levene_test(in_data,categorical_variable)
        # var.test()
    }else if(grepl('student',method)){
        # code unpaired vs paired t test in the future
        output=multiple_unpaired_t_test(in_data,categorical_variable,homo_variables,hetero_variables)
    }else if(grepl('kruskal',method)){
        return(error_massage)
    }else if(grepl('wilcoxon|manwhitney',method)){
        return(error_massage)
    }else if(grepl('anova|aov',method)){
        output=multiple_anova_test(in_data,categorical_variable)
    }else if(grepl('cor',method)){
        output=multiple_correlation_test(in_data,in_numeric_variable)
    }else{
        return(error_massage)
    }
    return(output)
}


getNumericSummaryTable=function(in_data,group_variable,summary_variable,set_color=color_function,...){
    # table
    temp<-in_data %>% 
    #group_by_at(vars(...)) %>% 
    group_by_at(vars(group_variable)) %>% 
    mutate(count=n())%>%
    summarise_at(vars(summary_variable,count),
                 list(mean=mean,
                 sd=sd,
                 min=min,
                 Q1=~quantile(., probs = 0.25),
                 median=median, 
                 Q3=~quantile(., probs = 0.75),
                 max=max))%>%
                 as.data.frame()%>%
                 rename(
                 n=count_mean)%>%
                 dplyr::select(-contains('count'))%>%
                 as.data.frame()
    names(temp)<-c("group",
    sapply(names(temp)[-1],function(x)str_replace(x,paste0(summary_variable,"_"),"")))
    output<-temp%>%
    mutate(
        variable=group_variable,
        summary=summary_variable,
        mean=mean%>%round(2),
        sd=sd%>%round(2),
        min=min%>%round(2),
        Q1=Q1%>%round(2),
        Q4=Q3%>%round(2),
        max=max%>%round(2),
        IQR_min=Q1-(Q3-Q1)*1.5%>%round(2),
    IQR_max=Q3+(Q3-Q1)*1.5%>%round(2),
    proportion=paste0(round(n/nrow(all_data)*100,2),"%"))%>%
    dplyr::select(variable,group,summary,n,proportion,mean,sd,min,IQR_min,Q1,median,Q3,IQR_max,max)
    return(output)
}

getNumericSummaryPlot=function(
    in_data=all_data,group_variable,summary_variable,
    set_color=color_function,
    summary_function=getNumericSummaryTable,...){
    # plot
    temp=getNumericSummaryTable(in_data,group_variable,summary_variable)
    temp2=temp
    names(temp2)[2]=group_variable
    plot<-
    in_data%>%
    dplyr::select(group_variable,summary_variable)%>%
    inner_join(.,temp2,by=group_variable)%>%
    ggplot(aes(x=age,fill=get(group_variable),color=get(group_variable)))+
    geom_histogram(aes(y=..density..),binwidth=1,alpha=0.5, position="identity")+
    geom_vline(aes(xintercept=mean,color=get(group_variable)), linetype="dashed", size=1.5) + 
    geom_density(aes(y=..density..),alpha=0.3) +
    scale_color_manual(values=set_color(nrow(temp2)))+
    scale_fill_manual(values=set_color(nrow(temp2)))+
    theme_bw()+
    theme(legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.margin = margin(6, 6, 6, 6),
    legend.text = element_text(size = 10))+
    guides(fill=guide_legend(title=group_variable),
    color=FALSE)+
    geom_text(aes(label=round(mean,1),y=0,x=mean),
                vjust=-1,col='yellow',size=5)+
    ggtitle(paste0("Histogram & Density, ", summary_variable, " Grouped by ", group_variable))+
        labs(x=summary_variable, y = "Density")

    result<-plot
    return(result)
}

age_data_generator=function(in_data,in_response,fun=scale_function){
    # this function generates a age (continuous) data that are statistically associated 
    # with a simulated variable as designed above.

    ## conduct t test with the response and each variable generated by multivariate normal distributions.
    ## search a variable with the largest difference in mean between the two groups or with the lowest p value
    ## In this case, I will pick the former one. 
    ## (I don't care about the multiple testing problems for now)
    temp_df=as.data.frame(matrix(ncol=5)) # initialize an empty data frame
    for(i in 1:ncol(in_data)){
        temp_df[i,]=c(
            names(in_data)[i],
            t.test(in_data[,i]~in_response)$estimate[1],
            t.test(in_data[,i]~in_response)$estimate[2],
            t.test(in_data[,i]~in_response)$estimate[2]-t.test(data[,i]~response)$estimate[1],
            t.test(in_data[,i]~in_response)$p.value)
    }
    names(temp_df)<-c('metabolite','mean_neg','mean_pos','mean_diff','p.value')

    ## search a variable with the largest difference in mean
    strong_metabolite<-
        temp_df%>%
        mutate(
            mean_neg=as.numeric(mean_neg),
            mean_pos=as.numeric(mean_pos),
            mean_diff=as.numeric(mean_diff),
            p.value=as.numeric(p.value),
            abs_mean_diff=abs(mean_diff))%>%
        filter(abs_mean_diff==max(abs_mean_diff))%>%
        dplyr::select(metabolite)%>%pull
    
    ## generate age data with min max normalization
    age_data<-
        data%>%
        dplyr::select(strong_metabolite)%>%
        scale_function(vector=.,min=65,max=105,method="customized")%>%
        rename(age=1)%>%round(0)
    return(age_data)
}


genotype_data_generator=function(in_response=response,fun=scale_function){
    # this function generates a genotype (categorical) data 
    # that are jointly and statistically associated with a continuous data and a binary data 
    # (I am not so sure if I can generate data that are statistically associated 
    # with some fake metabolite data. But, I will give it a try).

    ## Declare the marginal proportions 
    ## for binary (affected vs unaffected) and a genotype (categorical) data, respectively

    ##the simulated proportion for the disease vs non-disease cases
    binary_proportion<-as.numeric(table(in_response)/sample_size) 
    genotype_proportion<-c(0.084,0.779,0.137) # the known proportion of APOE genotypes from Wiki

    ## Declare the joint proportion predictor matrix
    joint_proportion<-matrix(
        c(binary_proportion[1]*genotype_proportion, # for the unaffected cases
        binary_proportion[2]*genotype_proportion),  # for the affected cases
        ncol=2, byrow=FALSE)

    # Generate the genotype (catogrical) data
    genotype_data = numeric(sample_size) # initialize a vector 
    for (i in 1:sample_size) {
        genotype_data[i] = sample(
            c('e2','e3','e4'),
             1, 
             prob=joint_proportion[,ifelse(grepl('neg',in_response[i]),1,2)])
    }    
    return(genotype_data)
}