
########### Library Diclaration ##################
library(ggplot2)
library(reshape2)
library(dplyr)
library(caret)
library(pROC)
library(flexclust)
library(e1071)
library(randomForest)

##########Reading File#################
pred_ren = read.csv("Renewal_Prediction.csv", header = TRUE, sep = ",")
###Summary
summary(pred_ren)


####new data frame for manipulation
new_pred_ren = pred_ren

#####Converting ? into NA's
#new_pred_ren[new_pred_ren == 0] = NA
new_pred_ren[new_pred_ren == '?'] = NA
#### In Renewal attribute 0's can't be treated at NA
new_pred_ren$Renewal = pred_ren$Renewal


#########Summary Again##############
summary(new_pred_ren)

#####drop columns having more than 40% NA's##############
#new_pred_ren_2 = new_pred_ren[ , !(names(new_pred_ren) %in% c("C11", "C15"))]
new_pred_ren_2 = new_pred_ren
#########Treatment###################
#Defactorised continuous variables
new_pred_ren_2$C2 = as.numeric(new_pred_ren_2$C2)
new_pred_ren_2$C14 = as.numeric(new_pred_ren_2$C14)

##Separting compete rows and rows having missing values
new_pred_ren_complete = (na.omit(new_pred_ren_2)) 
new_pred_ren_NAs =  new_pred_ren_2[-which(rownames(new_pred_ren_2)
                                          %in% rownames(new_pred_ren_complete)), ]

####drop rows which have NAs as they are just 5%

#################Univariate Analysis###########################
continuous_vars = c("C2", "C3", "C8", "C11", "C14", "C15")
categorical_vars = names(new_pred_ren_complete[, -which (names(new_pred_ren_complete) 
                                                         %in% continuous_vars)])
categorical_vars = categorical_vars[-10]
#########Continuous - Hist, Box
plot(new_pred_ren_complete$C2)
plot(new_pred_ren_complete$C3)
plot(new_pred_ren_complete$C8)
plot(new_pred_ren_complete$C11)
plot(new_pred_ren_complete$C14)
plot(new_pred_ren_complete$C15)

###Histogram
#melted = melt(new_pred_ren_complete, new_pred_ren_complete.var = "Label")
#ggplot(data = melted, aes(x=variable, y=value)) + geom_boxplot(aes(fill=Label))

###Boxplot
#boxplot(new_pred_ren_complete[, c("C2", "C3", "C8", "C11", "C14", "C15")])
boxplot(new_pred_ren_complete$C2)
boxplot(new_pred_ren_complete$C3)
boxplot(new_pred_ren_complete$C8)
boxplot(new_pred_ren_complete$C11)
boxplot(new_pred_ren_complete$C14)
boxplot(new_pred_ren_complete$C15)

###Categorical 
summary(new_pred_ren_complete)

#################multivariate Analysis###########################
#######Continuous - Continuos
pairs(new_pred_ren_complete[, c("C2", "C3", "C8", "C11", "C14", "C15")])

##correlation
correlation_tab = cor(new_pred_ren_complete[,continuous_vars])

######Categorical - Categorical
###drop unused levesl from the dataset
pred_ren_drpLevels = droplevels(new_pred_ren_complete)

chi_sqr_mat = matrix(nrow=9, ncol=9)
rownames(chi_sqr_mat) = paste(categorical_vars)
colnames(chi_sqr_mat) = paste(categorical_vars)

p_val_mat = matrix(nrow=9, ncol=9)
rownames(p_val_mat) = paste(categorical_vars)
colnames(p_val_mat) = paste(categorical_vars)
#######chi - square test##########
for (x in (1:8)) {
  for (y in ((x+1):9)) {
    p_val_mat[x,y] = as.numeric(summary(xtabs(~categorical_vars[x] + categorical_vars[y] , 
                                   data = pred_ren_drpLevels))["p.value"])
    chi_sqr_mat[x,y]= as.numeric(summary(xtabs(~categorical_vars[x] + categorical_vars[y] , 
                                    data = pred_ren_drpLevels))["statistic"])
  }
  
}

summary(xtabs(~C9 + C10 , data = pred_ren_drpLevels))
summary(xtabs(~C9 + C12 , data = pred_ren_drpLevels))
summary(xtabs(~C10 + C12 , data = pred_ren_drpLevels))

##########Continuous - Categorical ####ANOVA test
##########continuous variable vs Renewal#########
pred_ren_drpLevels$Renewal = as.factor(pred_ren_drpLevels$Renewal)


###############Outlier detection, analysis and treatment###############
#########use only continuos variables#######################
outlier_df = setNames(data.frame(matrix(ncol = 16, nrow = 0)), colnames(pred_ren_drpLevels))
outlier_df = rbind(pred_ren_drpLevels[pred_ren_drpLevels$C3 %in% (boxplot.stats(pred_ren_drpLevels$C3)$out),],
                   pred_ren_drpLevels[pred_ren_drpLevels$C8 %in% (boxplot.stats(pred_ren_drpLevels$C8)$out),],
                   pred_ren_drpLevels[pred_ren_drpLevels$C11 %in% (boxplot.stats(pred_ren_drpLevels$C11)$out),],
                   pred_ren_drpLevels[pred_ren_drpLevels$C14 %in% (boxplot.stats(pred_ren_drpLevels$C14)$out),],
                  pred_ren_drpLevels[pred_ren_drpLevels$C15 %in% (boxplot.stats(pred_ren_drpLevels$C15)$out),])

outlier_df = unique(outlier_df)  ###184
########log transform###############
log_trans = pred_ren_drpLevels
log_trans$C3 = log(log_trans$C3 + 0.0000001)
log_trans$C8 = log(log_trans$C8+ 0.0000001)
log_trans$C11 = log(log_trans$C11+ 0.0000001)
log_trans$C15 = log(log_trans$C15+ 0.0000001)

#######
outlier_df_lg = setNames(data.frame(matrix(ncol = 16, nrow = 0)), colnames(log_trans))
outlier_df_lg = rbind(log_trans[log_trans$C3 %in% (boxplot.stats(log_trans$C3)$out),],
                   log_trans[log_trans$C8 %in% (boxplot.stats(log_trans$C8)$out),],
                   log_trans[log_trans$C11 %in% (boxplot.stats(log_trans$C11)$out),],
                   log_trans[log_trans$C14 %in% (boxplot.stats(log_trans$C14)$out),],
                   log_trans[log_trans$C15 %in% (boxplot.stats(log_trans$C15)$out),])

outlier_df_lg = unique(outlier_df_lg) ##73
###############Feature Engineering################
####Binary Level encoding for categorical variables#######

pred_ren_encoded = pred_ren_drpLevels
pred_ren_encoded$C1 = as.numeric(pred_ren_encoded$C1)
pred_ren_encoded$C4 = as.numeric(pred_ren_encoded$C4)
pred_ren_encoded$C5 = as.numeric(pred_ren_encoded$C5)
pred_ren_encoded$C6 = as.numeric(pred_ren_encoded$C6)
pred_ren_encoded$C7 = as.numeric(pred_ren_encoded$C7)
pred_ren_encoded$C9 = as.numeric(pred_ren_encoded$C9)
pred_ren_encoded$C10 = as.numeric(pred_ren_encoded$C10)
pred_ren_encoded$C12 = as.numeric(pred_ren_encoded$C12)
pred_ren_encoded$C13 = as.numeric(pred_ren_encoded$C13)

###for log transformed data
log_trans_encoded = log_trans
log_trans_encoded$C1 = as.numeric(pred_ren_encoded$C1)
log_trans_encoded$C4 = as.numeric(pred_ren_encoded$C4)
log_trans_encoded$C5 = as.numeric(pred_ren_encoded$C5)
log_trans_encoded$C6 = as.numeric(pred_ren_encoded$C6)
log_trans_encoded$C7 = as.numeric(pred_ren_encoded$C7)
log_trans_encoded$C9 = as.numeric(pred_ren_encoded$C9)
log_trans_encoded$C10 = as.numeric(pred_ren_encoded$C10)
log_trans_encoded$C12 = as.numeric(pred_ren_encoded$C12)
log_trans_encoded$C13 = as.numeric(pred_ren_encoded$C13)

####converting decimal into binary


###############statical test and analysis##############
############ t-test #############
t_stats_df = data.frame(matrix(NA, nrow=15, ncol=2))
names(t_stats_df) = c("fields", "p_value")
t_stats_df$fields = names(pred_ren_encoded[-16])
t_stats_df$p_value = lapply(pred_ren_encoded[-16], function(x) t.test(x ~ pred_ren_encoded$Renewal)["p.value"])
t_stats_df$p_value = as.numeric(unlist(t_stats_df$p_value))

###after log tranform
t_stats_log = data.frame(matrix(NA, nrow=15, ncol=2))
names(t_stats_log) = c("fields", "p_value")
t_stats_log$fields = names(log_trans_encoded[-16])
t_stats_log$p_value = lapply(log_trans_encoded[-16], function(x) t.test(x ~ log_trans_encoded$Renewal)["p.value"])
t_stats_log$p_value = as.numeric(unlist(t_stats_log$p_value))

####storing significant variables
significant_vars =  t_stats_log[t_stats_log$p_value < 0.05,"fields"]
insignificant_vars = t_stats_log[t_stats_log$p_value > 0.05,"fields"]

########################Modelling ##############################
dt = sort(sample(nrow(log_trans_encoded), nrow(log_trans_encoded)*.8))
train<-log_trans_encoded[dt,]
test<-log_trans_encoded[-dt,]

############logistic regression#############
model_logistic <- glm (Renewal ~ .-Renewal, data = train, family = binomial)
summary(model_logistic)

predict <- predict(model_logistic, type = 'response', newdata = test[,-16])
predict = ifelse(predict > 0.5,1,0)

confusionMatrix(predict, test$Renewal)
roc(test$Renewal, predict)

####with fields (p<0.05) ##########
model_logreg_sig = glm (Renewal ~ .-Renewal, data = train[, !colnames(train) %in% 
                                                   insignificant_vars], family = binomial)
predict_logreg_sig = predict(model_logreg_sig, type = 'response', 
                             newdata = test[,significant_vars])
predict_logreg_sig = ifelse(predict_logreg_sig > 0.5,1,0)

confusionMatrix(predict_logreg_sig, test$Renewal)
roc(test$Renewal, predict_logreg_sig)

###########K-means################
#model_kmeans = kcca(train, k=2, kccaFamily("kmeans"))
#predict_kmeans <- predict(model_kmeans, newdata=test, k=2, kccaFamily("kmeans"))

###########SVM #################
model_svm = svm(Renewal ~ .-Renewal, data = train, kernel = "linear", gamma = 1, cost = 2,type="C-classification")
predict_svm = predict(model_svm, newdata = test[-16])

confusionMatrix(predict_svm, test$Renewal)
roc(test$Renewal, as.numeric(predict_svm) - 1)

####with fields (p<0.05) ##########
model_svm_sig = svm(Renewal ~ .-Renewal, data = train[, !colnames(train) %in% insignificant_vars], 
                    kernel = "linear", gamma = 1, cost = 2,type="C-classification")
predict_svm_sig = predict(model_svm_sig, newdata = test[, significant_vars])

confusionMatrix(predict_svm_sig, test$Renewal)
roc(test$Renewal, as.numeric(predict_svm_sig) - 1)


###########Random Forest##########
model_rf = randomForest(Renewal ~ .-Renewal, data = train, ntree = 1000)
predict_rf = predict(model_rf, newdata = test[-16])

confusionMatrix(predict_rf, test$Renewal)
roc(test$Renewal, as.numeric(predict_rf) - 1)

####with fields (p<0.05) ##########
model_rf_sig = randomForest(Renewal ~ .-Renewal, data = train[, !colnames(train) %in% insignificant_vars],
                            ntree = 1000)
predict_rf_sig = predict(model_rf_sig, newdata = test[, significant_vars])

confusionMatrix(predict_rf_sig, test$Renewal)
roc(test$Renewal, as.numeric(predict_rf_sig) - 1)

