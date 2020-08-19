## ----general-options,echo=FALSE-----------------------------------

library(knitr)
# output code, but no warnings
opts_chunk$set(echo = TRUE,eval=TRUE,warning=FALSE,cache=TRUE)
# auto check dependencies (of cached chunks, its an approximation only)
opts_chunk$set(autodep = TRUE)
# dep_auto() # print dependencies 



## ----load-packages,message=FALSE----------------------------------
# install.packages(c("grpgreg", "glmnet", "kernlab", "caret", "ranger", "mboost",
#                    "gbm", "geoGAM", "raster"))
library(grpreg) # for grouped lasso
library(glmnet) # for general lasso
library(kernlab) # for support vector machines
library(caret) # for model tuning
library(ranger) # to fit random forest
library(mboost) # for the boosting models with linear and spline terms  
library(gbm) # for the boosting model with trees
library(geoGAM) # for the berne dataset
library(raster) # for plotting as a raster
library(parallel) # for parallel computing


## ----read-in-data-------------------------------------------------
data(berne)
dim(berne)
# Continuous response 
d.ph10 <- berne[berne$dataset == "calibration" & !is.na(berne$ph.0.10), ]
d.ph10 <- d.ph10[complete.cases(d.ph10[13:ncol(d.ph10)]), ]
# select validation data for subsequent validation
d.ph10.val <- berne[berne$dataset == "validation" & !is.na(berne$ph.0.10), ]
d.ph10.val <- d.ph10.val[complete.cases(d.ph10.val[13:ncol(d.ph10)]), ]
# Binary response 
d.wlog100 <- berne[berne$dataset=="calibration"&!is.na(berne$waterlog.100), ]
d.wlog100 <- d.wlog100[complete.cases(d.wlog100[13:ncol(d.wlog100)]), ]
# Ordered/multinomial tesponse 
d.drain <- berne[berne$dataset == "calibration" & !is.na(berne$dclass), ]
d.drain <- d.drain[complete.cases(d.drain[13:ncol(d.drain)]), ]
# covariates start at col 13
l.covar <- names(d.ph10[, 13:ncol(d.ph10)])


## ----apply-example------------------------------------------------
# loop 
# first create a vector to save the results
t.result <- c()
for( ii in 1:10 ){ t.result <- c(t.result, ii^2) }
# the same as apply
t.result <- sapply(1:10, function(ii){ ii^2 })
# of course, this example is even shorter using:
t.result <- (1:10)^2


## ----lasso-continuous-response,cache=TRUE-------------------------

# define groups: dummy coding of a factor is treated as group
# find factors
l.factors <- names(d.ph10[l.covar])[ 
  t.f <- unlist( lapply(d.ph10[l.covar], is.factor) ) ]
l.numeric <-  names(t.f[ !t.f ])

# create a vector that labels the groups with the same number  
#  each numeric has its own number
#  all dummy variables of a factor go into one group and have the same number
g.groups <- c( 1:length(l.numeric), 
               unlist( 
                 sapply(1:length(l.factors), function(n){
                   rep(n+length(l.numeric), nlevels(d.ph10[, l.factors[n]])-1)
                 }) 
               ) 
)
# grpreg needs model matrix as input
#  this creates dummy covariates, 
#  without an intercept (-1) as it is added in grpreg
XX <- model.matrix( ~., d.ph10[, c(l.numeric, l.factors)])[,-1]

# cross validation (CV) to find lambda
ph.cvfit <- cv.grpreg(X = XX, y = d.ph10$ph.0.10, 
                      group = g.groups, 
                      penalty = "grLasso",
                      returnY = T) # access CV results


## ----lasso-predictions--------------------------------------------
# choose optimal lambda: CV minimum error + 1 SE (see glmnet)
l.se <- ph.cvfit$cvse[ ph.cvfit$min ] + ph.cvfit$cve[ ph.cvfit$min ]
idx.se <- min( which( ph.cvfit$cve < l.se ) ) - 1

# create model matrix for validation set
newXX <- model.matrix( ~., d.ph10.val[, c(l.factors, l.numeric), F])[,-1]
t.pred.val <-  predict(ph.cvfit, X = newXX, 
                       type = "response",
                       lambda =  ph.cvfit$lambda[idx.se])
# get CV predictions, e.g. to compute R2
ph.lasso.cv.pred <- ph.cvfit$Y[,idx.se]


## ----lasso-get-model----------------------------------------------
# get the non-zero coefficients:
t.coef <- ph.cvfit$fit$beta[, idx.se ]
t.coef[ t.coef > 0 ]


## ----lasso-plot-cv,echo=FALSE,fig.width=7,fig.height=4.5, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Cross validation error plotted against the tuning parameter lambda. The dashed line indicates lambda at minimal error, the dotted darkgrey line is the optimal lambda with minimal error + 1 SE."----
plot(ph.cvfit)
abline( h = l.se, col = "grey", lty = "dotted")
abline( v = log( ph.cvfit$lambda[ idx.se ]), col = "grey30", lty = "dotted")


## ----lasso-multinomial-response,cache = TRUE----------------------

# create model matrix for drainage classes
# use a subset of covariates only, because model optimization for 
# multinomial takes long otherwise

set.seed(42) # makes sample() reproducible
XX <- model.matrix(~.,d.drain[, l.covar[sample(1:length(l.covar), 30)]])[,-1]

drain.cvfit <- cv.glmnet( XX, d.drain$dclass, nfold = 10,  
                          keep = T, # access CV results
                          family = "multinomial", 
                          type.multinomial = "grouped")


## ----lasso-multinomial-response-coeffs,cache=TRUE-----------------

drain.fit <- glmnet( XX, d.drain$dclass,
                     family = "multinomial", 
                     type.multinomial = "grouped",
                     lambda = drain.cvfit$lambda.min)
# The coeffs are here:
# drain.fit$beta$well
# drain.fit$beta$moderate
# drain.fit$beta$poor


## ----svm,cache=TRUE-----------------------------------------------

# We have to set up the design matrix ourselfs 
# (without intercept, hence remove first column) 
XX <- model.matrix( ~., d.ph10[, c(l.covar), F])[,-1]

# set seed for random numbers to split cross-valiation sets
set.seed(31)
# Setup for 10fold cross-validation
ctrl <- trainControl(method = "cv",   
                     number = 10,		   
                     savePredictions = "final")

# 1. pass of training - find region of C and lambda 
svm.tune1 <- train(x = XX,
                   y = d.ph10[, "ph.0.10"],
                   method = "svmRadial", # radial kernel function
                   tuneLength = 9, # check 9 values of the cost function
                   preProc = c("center","scale"), # center and scale data
                   trControl=ctrl)

# 2. pass of training - find best value for C and lambda
# setup a tuning grid with values around the result of the first pass
sig <- svm.tune1$bestTune$sigma
t.sigma <- sort( unique( round(abs( c(sig, sig + seq(0, sig*2, by = sig/1), 
                                      sig - seq(0, sig*2, by = sig/1)) ), 6)))
tune.grid <- expand.grid(
  sigma = t.sigma[t.sigma>0], # sigma must be positive
  C = sort( unique( abs( c(svm.tune1$bestTune$C, 
                           svm.tune1$bestTune$C - seq(0, 0.3, by = 0.1), 
                           svm.tune1$bestTune$C + seq(0, 0.3, by = 0.1) )) ))
)
# Train and Tune the SVM
svm.model <- train(x = XX,
                   y = d.ph10[, "ph.0.10"],
                   method = "svmRadial",
                   preProc = c("center","scale"),
                   tuneGrid = tune.grid,
                   trControl = ctrl)
# -> if this takes too long: take a short cut with
# svm.model <- svm.tune1


## ----svm-validation-plots,fig.width=10,fig.height=5, fig.align='center', out.width='0.85\\textwidth',fig.cap = "Predictions from cross-validation (left) and the validation dataset (right) plotted against the observed values (dashed: 1:1-line, green: lowess scatterplott smoother)."----
# create validation plots with lowess scatterplot smoothers
# for cross-validation
par(mfrow = c(1,2))
plot(svm.model$pred$pred, svm.model$pred$obs,  
     xlab = "cross-validation predictions", 
     ylab = "observed", 
     asp = 1)
abline(0,1, lty = "dashed", col = "grey")
lines(lowess(svm.model$pred$pred, svm.model$pred$obs), col = "darkgreen", lwd = 2)

# for independent validation set
# calculate predictions for the validation set
newXX <- model.matrix( ~., d.ph10.val[, l.covar, F])[,-1]
t.pred.val <- predict.train(svm.model, newdata = newXX)
plot(t.pred.val, d.ph10.val[, "ph.0.10"], 
     xlab = "predictions on validation set", 
     ylab = "observed", 
     asp = 1)
abline(0,1, lty = "dashed", col = "grey")
lines(lowess(t.pred.val, d.ph10.val[, "ph.0.10"]), col = "darkgreen", lwd = 2)


## ----random-forest,cache=TRUE-------------------------------------

# Fit a random forest with default parameters 
# (often results are already quite good)
set.seed(1)
rf.model.basic <- ranger(x = d.ph10[, l.covar ],
                         y = d.ph10[, "ph.0.10"])

# tune main tuning parameter "mtry"
# (the number of covariates that are randomly selected to try at each split)

# define function to use below
f.tune.randomforest <- function(test.mtry, # mtry to test
                                d.cal,     # calibration data.frame
                                l.covariates ){ # list of covariates
  # set seed 
  set.seed(1)
  # fit random forest with mtry = test.mtry
  rf.tune <- ranger(x = d.cal[, l.covariates ],
                    y = d.cal[, "ph.0.10"],
                    mtry = test.mtry)
  # return the mean squared error (mse) of this model fit
  return( rf.tune$prediction.error ) 
}

# vector of mtry to test 
seq.mtry <- 1:(length(l.covar) - 1)
# Only take every fifth mtry to speed up tuning
seq.mtry <- seq.mtry[ seq.mtry %% 5 == 0 ] 

# Apply function to sequence. 
# (without parallel computing this takes a while)
t.OOBe <-   mclapply(seq.mtry, # give sequence 
                     FUN = f.tune.randomforest, # give function name
                     mc.cores = 1, ## number of CPUs 
                     mc.set.seed = FALSE, # do not use new seed each time 
                     # now here giv the arguments to the function:
                     d.cal = d.ph10, 
                     l.covar = l.covar ) 

# Hint: Who is not comfortable with "mclapply" 
# the same could be achieved with
# for(test.mtry in 1:m.end){ 
#    .. content of function + vector to collect result... }

# create a dataframe of the results           
mtry.oob <- data.frame(mtry.n = seq.mtry, mtry.OOBe = unlist(t.OOBe))

# get the mtry with the minimum MSE
s.mtry <- mtry.oob$mtry.n[ which.min(mtry.oob$mtry.OOBe) ]

# compute random forest with optimal mtry 
set.seed(1)
rf.model.tuned <- ranger(x = d.ph10[, l.covar ],
                         y = d.ph10[, "ph.0.10"],
                         mtry = s.mtry)


## ----random-forest-plot-mtry,fig.width=6,fig.height=4.7, fig.align='center',fig.pos='!h',out.width='0.6\\textwidth',fig.cap = "Tuning parameter mtry plotted against the out-of-bag mean squared error (grey line: lowess smoothing line, dashed line: mtry at minimum MSE)."----
plot( mtry.oob$mtry.n, mtry.oob$mtry.OOBe, pch = 4, 
      ylab = "out-of-bag MSE error", xlab = "mtry")
abline(v = s.mtry, lty = "dashed", col = "darkgrey")
lines( lowess( mtry.oob$mtry.n, mtry.oob$mtry.OOBe ), lwd = 1.5, col = "darkgrey")


## ----boosted-trees-tuning,cache=TRUE------------------------------

# create a grid of the tuning parameters to be tested, 
# main tuning parameters are: 
gbm.grid <- expand.grid(
  # how many splits does each tree have
  interaction.depth = c(2,5,10),
  # how many trees do we add (number of iterations of boosting algorithm)
  n.trees = seq(2,100, by = 5),
  # put the shrinkage factor to 0.1 (=10% updates as used 
  # in package mboost), the default (0.1%) is a bit too small, 
  # makes model selection too slow. 
  # minimum number of observations per node can be left as is
  shrinkage = 0.1, n.minobsinnode = 10) 

# make tuning reproducible (there are random samples for the cross validation)
set.seed(291201945)

# Train the gbm model 
#   Remove "ge_caco3" throws an error since Package gbm 2.1.5,
#   this bug is reported: https://github.com/gbm-developers/gbm/issues/40
gbm.model <- train(x = d.ph10[, l.covar[-c(50)]], 
                   y = d.ph10[, "ph.0.10"],
                   method = "gbm", # choose "generalized boosted regression model"
                   tuneGrid = gbm.grid,
                   verbose = FALSE,
                   trControl = trainControl(
                     # use 10fold cross validation (CV)
                     method = "cv", number = 10,
                     # save fitted values (e.g. to calculate RMSE of the CV)
                     savePredictions = "final"))

# print optimal tuning parameter
gbm.model$bestTune


## ----boosted-trees-map,fig.width=5,fig.height=5, fig.align='center', out.width='0.7\\textwidth',fig.cap = "Predictions computed with an optimized boosted trees model of topsoil pH (0--10 cm) for a very small part of the Berne study region (white areas are streets, developped areas or forests, CRAN does not accept larger datasets)."----

# compute predictions for the small part of the study area
# (agricultural land, the empty pixels are streets, forests etc.)
data("berne.grid")

berne.grid$pred <- predict.train(gbm.model, newdata = berne.grid )

# create a spatial object for a proper spatial plot
coordinates(berne.grid) <- ~x+y
# add the Swiss projection (see ?berne.grid)
# see https://epsg.io for details on projections
proj4string(berne.grid) <- CRS("+init=epsg:21781")
# create a raster object from the spatial point dataframe 
gridded(berne.grid) <- TRUE
plot(raster(berne.grid, layer = "pred"))



## ----boosted-trees-partial-dependencies,fig.pos="h",fig.width=7,fig.height=7, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Partial dependence plots of boosted trees model for the four most important covariates."----

# get variable importance
t.imp <- varImp(gbm.model$finalModel) 

# check how many covariates were never selected
sum( t.imp$Overall == 0 )

# order and select 4 most important covariates
t.names <- dimnames(t.imp)[[1]][ order(t.imp$Overall, decreasing = T)[1:4] ]

par(mfrow = c(2,2))
for( name in t.names ){
  # select index of covariate
  ix <- which( gbm.model$finalModel$var.names == name )
  plot(gbm.model$finalModel, i.var = ix)
}

# -> improve the plots by using the same y-axis (e.g. ylim=c(..,..)) 
#    for all of them, and try to add labels (xlab = , ylab = ) 
#    or a title (main = )



## ----glmboost,cache=TRUE------------------------------------------
# Fit model
ph.glmboost <- glmboost(ph.0.10 ~., data = d.ph10[ c("ph.0.10", l.covar)],
                        control = boost_control(mstop = 200),
                        center = TRUE)

# Find tuning parameter: mstop = number of boosting itertations
set.seed(42)
ph.glmboost.cv <- cvrisk(ph.glmboost, 
                         folds = mboost::cv(model.weights(ph.glmboost), 
                                            type = "kfold"))

# print optimal mstop
mstop(ph.glmboost.cv)

## print model with fitted coefficents 
# ph.glmboost[ mstop(ph.glmboost.cv)]


## ----glmboost-plot,fig.width=7,fig.height=5, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Path of cross validation error along the boosting iterations.", echo = FALSE----
plot(ph.glmboost.cv)


## ----gamboost,cache=TRUE,message=FALSE----------------------------

# quick set up formula

# Response
f.resp <- "ph.0.10 ~ "

# Intercept, add to dataframe 
f.int <- "bols(int, intercept = F, df = 1)"
d.ph10$int <- rep(1, nrow(d.ph10))

# Smooth spatial surface (needs > 4 degrees of freedom)
f.spat <- "bspatial(x, y, df = 5, knots = 12)"

# Linear baselearners for factors, maybe use df = 5
f.fact <- paste( 
  paste( "bols(", l.factors, ", intercept = F)" ), 
  collapse = "+" 
)

# Splines baselearners for continuous covariates
f.num <- paste( 
  paste( "bbs(", l.numeric, ", center = T, df = 5)" ),
  collapse = "+"
)

# create complete formula 
ph.form <- as.formula( paste( f.resp, 
                              paste( c(f.int, f.num, f.spat, f.fact),
                                     collapse = "+")) ) 
# fit the boosting model
ph.gamboost  <- gamboost(ph.form, data = d.ph10,
                         control = boost_control(mstop = 200))

# Find tuning parameter
ph.gamboost.cv <- cvrisk(ph.gamboost, 
                         folds = mboost::cv(model.weights(ph.gamboost), 
                                            type = "kfold"))


## ----gamboost-results---------------------------------------------
# print optimal mstop
mstop(ph.gamboost.cv)

## print model info 
ph.gamboost[ mstop(ph.glmboost.cv)]
## print number of chosen baselearners 
length( t.sel <-  summary( ph.gamboost[ mstop(ph.glmboost.cv)] )$selprob ) 

# Most often selected were: 
summary( ph.gamboost[ mstop(ph.glmboost.cv)] )$selprob[1:5]  


## ----gamboost-partial-plots,echo=FALSE,fig.width=7,fig.height=6, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Residual plots of the 4 covariates with highest selection frequency."----
par(mfrow=c(2,2) )
plot(ph.gamboost[ mstop(ph.glmboost.cv)], which = names(t.sel[1:4]) )


## ----gamboost-partial-plots-spatial,echo=FALSE,fig.width=7,fig.height=5, fig.align='center', out.width='0.8\\textwidth',fig.cap = "Modelled smooth spatial surface based on the coordinates."----
par(mfrow=c(1,1) )
plot(ph.gamboost[ mstop(ph.glmboost.cv)], which = grep("bspat", names(t.sel), value = T) )


## ----session-info,results='asis'----------------------------------
toLatex(sessionInfo(), locale = FALSE)


## ----export-r-code,echo=FALSE,result="hide"-----------------------
# purl("OpenGeoHub-machine-learning-training-1.Rnw")

