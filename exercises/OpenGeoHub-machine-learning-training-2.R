## ----general-options,echo=FALSE-----------------------------------
# This code is used to generate the PDF (knitr report)
library(knitr)
# output code, but no warnings
opts_chunk$set(echo = TRUE,eval=TRUE,warning=FALSE)
# auto check dependencies (of cached chunks, its an approximation only)
opts_chunk$set(autodep = TRUE)
# dep_auto() # print dependencies 


## ----load-packages,message=FALSE----------------------------------
library(randomForest) # for random forest models
library(quantregForest) # for quantile random forest
library(grpreg) # for group lasso
library(geoGAM) # for the Berne test data set


## ----read-in-data-------------------------------------------------
dim(berne)
# Select soil pH in 0-10 cm as continuous response, 
# select calibration data and remove rows with missing pH 
d.ph10 <- berne[ berne$dataset == "calibration" & !is.na(berne$ph.0.10), ]
d.ph10 <- d.ph10[ complete.cases(d.ph10[13:ncol(d.ph10)]), ]
# covariates start at col 13
l.covar <- names(d.ph10[, 13:ncol(d.ph10)])


## ----fit-random-forest,cache=TRUE---------------------------------
set.seed(17)
rf.ph <- randomForest(x = d.ph10[, l.covar],
                      y = d.ph10$ph.0.10)


## ----plot-covar-importance, fig.width=5, fig.height=6, fig.align='center', fig.pos="!hb", out.width='0.5\\textwidth', fig.cap="Covariate importance of 20 most important covariates for topsoil pH (before selection)."----
varImpPlot(rf.ph, n.var = 20, main = "")


## ----select-random-forest,cache=TRUE------------------------------
# speed up the process by removing 5-10 covariates at a time
s.seq <- sort( c( seq(5, 95, by = 5), 
                  seq(100, length(l.covar), by = 10) ), 
               decreasing = T)

# collect results in list
qrf.elim <- oob.mse <- list()

# save model and OOB error of current fit        
qrf.elim[[1]] <- rf.ph
oob.mse[[1]] <- tail(qrf.elim[[1]]$mse, n=1)
l.covar.sel <- l.covar

# Iterate through number of retained covariates           
for( ii in 1:length(s.seq) ){
  t.imp <- importance(qrf.elim[[ii]], type = 2)
  t.imp <- t.imp[ order(t.imp[,1], decreasing = T),]
  
  qrf.elim[[ii+1]] <- randomForest(x = d.ph10[, names(t.imp[1:s.seq[ii]])],
                                   y = d.ph10$ph.0.10 )
  oob.mse[[ii+1]] <- tail(qrf.elim[[ii+1]]$mse,n=1)
  
}


# Prepare a data frame for plot
elim.oob <- data.frame(elim.n = c(length(l.covar), s.seq[1:length(s.seq)]), 
                       elim.OOBe = unlist(oob.mse) )


## ----plot-selection-path,fig.align='center',echo=FALSE,fig.height = 5,out.width='0.8\\textwidth',fig.cap = "Path of out-of-bag mean squared error as covariates are removed. Minimum is found at 55 covariates."----

plot(elim.oob$elim.n, elim.oob$elim.OOBe, 
     ylab = "OOB error (MSE)",
     xlab = "n covariates", 
     pch = 20)
abline(v = elim.oob$elim.n[ which.min(elim.oob$elim.OOBe)], lty = "dotted")


## ----partial-residual-plots-lm-lasso,fig.width=7,fig.height=4, fig.align='center', out.width='0.9\\textwidth',fig.cap = "Partial residual plots for a climate covariate in the ordinary least squares fit and the lasso."----
# create a linear model (example, with covariates from lasso)
ols <- lm( ph.0.10 ~ timeset + ge_geo500h3id + cl_mt_gh_4 + 
             tr_se_curvplan2m_std_25c, data = d.ph10 ) 
par(mfrow = c(1,2)) # two plots on same figure
# residual plot for covariate cl_mt_gh_4
termplot(ols, partial.resid = T, terms = "cl_mt_gh_4",
         ylim = c(-2,2),
         main = "Ordinary Least Squares")
abline(h=0, lty = 2)

## Create partial residual plot for lasso 
# there is no direct function available, but we can easily 
# construct the plot with
# y-axis: residuals + effect of term (XBi), scaled
# x-axis: values covariate
# regression line: model fit of axis y~x 

## First setup and fit the model 
l.factors <- names(d.ph10[l.covar])[ 
  t.f <- unlist( lapply(d.ph10[l.covar], is.factor) ) ]
l.numeric <-  names(t.f[ !t.f ])
# create a vector that labels the groups with the same number  
g.groups <- c( 1:length(l.numeric), 
               unlist( 
                 sapply(1:length(l.factors), function(n){
                   rep(n+length(l.numeric), nlevels(d.ph10[, l.factors[n]])-1)
                 }) 
               ) 
)
# grpreg needs model matrix as input
XX <- model.matrix( ~., d.ph10[, c(l.numeric, l.factors), F])[,-1]
# cross validation (CV) to find lambda
ph.cvfit <- cv.grpreg(X = XX, y = d.ph10$ph.0.10, 
                      group = g.groups, 
                      penalty = "grLasso",
                      returnY = T) # access CV results
# choose optimal lambda: CV minimum error + 1 SE (see glmnet)
l.se <- ph.cvfit$cvse[ ph.cvfit$min ] + ph.cvfit$cve[ ph.cvfit$min ]
idx.se <- min( which( ph.cvfit$cve < l.se ) ) - 1

# get the non-zero coefficients:
t.coef <- ph.cvfit$fit$beta[, idx.se ]

# get the index of the covariate
idx <- which( names(t.coef) == "cl_mt_gh_4" )

# residuals of lasso model chosen above
residuals <- d.ph10$ph.0.10 - ph.cvfit$Y[,idx.se] 
# prediction for this covariate XBi
Xbeta <- ph.cvfit$fit$beta[idx, idx.se] * d.ph10$cl_mt_gh_4
# calculate partial residuals and center with mean
part.resid <- scale(residuals + Xbeta, scale = F)[,1]

# plot with similar settings
plot(d.ph10$cl_mt_gh_4, 
     part.resid, pch = 1, col = "grey",
     ylim = c(-2,2),
     ylab = "partial residuals [%]", xlab = "cl_mt_gh_4",
     main = "Lasso")
abline(lm(part.resid ~ d.ph10$cl_mt_gh_4), col = "red")
abline(h=0, lty = 2)


## ----partial-dep-rf,fig.width=7,fig.height=8, fig.align='center', out.width='0.9\\textwidth',fig.cap = "Partial dependence plots for the 4 most important covariates."----
# select the model with minimum OOB error
rf.selected <- qrf.elim[[ which.min(elim.oob$elim.OOBe)]]

t.imp <- importance(rf.selected, type = 2)
t.imp <- t.imp[ order(t.imp[,1], decreasing = T),]

# 4 most important covariates
( t.3 <- names( t.imp[ 1:4 ] ) )

par( mfrow = c(2,2))

# Bug in partialPlot(): function does not allow a variable for the 
#  covariate name (e. g. x.var = name) in a loop
partialPlot(x = rf.selected, 
            pred.data = d.ph10[, names(rf.selected$forest$xlevels)], 
            x.var = "cl_mt_rr_3", ylab = "ph [-]", main = "") 
partialPlot(x = rf.selected, 
            pred.data = d.ph10[, names(rf.selected$forest$xlevels)], 
            x.var = "cl_mt_rr_11", ylab = "ph [-]", main = "" ) 
partialPlot(x = rf.selected, 
            pred.data = d.ph10[, names(rf.selected$forest$xlevels)], 
            x.var = "timeset", ylab = "ph [-]", main = "" ) 
partialPlot(x = rf.selected, 
            pred.data = d.ph10[, names(rf.selected$forest$xlevels)], 
            x.var = "cl_mt_rr_y", ylab = "ph [-]", main = "" ) 



## ----quantRF,cache=TRUE-------------------------------------------
# Fit quantile regression forest 
ph.quantRF <- quantregForest(x = d.ph10[, l.covar[1:30]],
                             y = d.ph10$ph.0.10) 

# select validation data
d.ph10.val <- berne[berne$dataset == "validation" & !is.na(berne$ph.0.10), ]
d.ph10.val <- d.ph10.val[complete.cases(d.ph10.val[l.covar]), ]

# compute predictions (mean) for each validation site
# (use function from random forest package)
ph.pred <- randomForest:::predict.randomForest(ph.quantRF,
                                               newdata = d.ph10.val)


## ----investigate-single-point,echo=FALSE,fig.pos='!h',fig.height=5,fig.width=4,fig.align='center', out.width='0.4\\textwidth',fig.cap= "Histogram of predictive distribution for one single prediction point (dotted lines: 90 \\% prediction interval, dashed line: mean prediction)."----

## predict 0.01, 0.02,..., 0.99 quantiles for validation data
ph.pred.distribution <- predict(ph.quantRF,
                                newdata = d.ph10.val, 
                                what = seq(0.01, 0.99, by = 0.01))

# plot predictive distribution for one site
sel.site <- 12
hist( ph.pred.distribution[sel.site,], 
      col = "grey", main = "",
      xlab = "predicted pH [-]", breaks = 12)

# add 90 % prediction interval to plot
abline(v = c( ph.pred.distribution[sel.site, "quantile= 0.05"],
              ph.pred.distribution[sel.site, "quantile= 0.95"]), 
       lty = "dotted")
abline(v = ph.pred[sel.site], lty = "dashed")


## ----create-intervall-plot,fig.height=5,fig.align='center',echo=FALSE, out.width='0.8\\textwidth',fig.cap= "Coverage of 90 \\%-prediction intervals computed by model-based boostrap."----

# get 90% quantiles for each point
t.quant90 <- cbind( 
  ph.pred.distribution[, "quantile= 0.05"],
  ph.pred.distribution[, "quantile= 0.95"])

# get index for ranking in the plot
t.ix <- sort( ph.pred, index.return = T )$ix

# plot predictions in increasing order
plot(
  ph.pred[t.ix], type = "n",
  ylim = range(c(t.quant90, ph.pred, d.ph10.val$ph.0.10)),
  xlab = "rank of predictions", 
  ylab =  "ph [-]" 
) 

# add prediction intervals
segments(
  1:nrow( d.ph10.val ),
  t.lower <- (t.quant90[,1])[t.ix],
  1:nrow( d.ph10.val ),
  t.upper <- (t.quant90[,2])[t.ix],
  col = "grey"
)

# select colour for dots outside of intervals
t.col <- sapply(
  1:length( t.ix ),
  function( i, x, lower, upper ){
    as.integer( cut( x[i], c( -Inf, lower[i]-0.000001, 
                              x[i], upper[i]+0.000001, Inf ) ) )
  },
  x = d.ph10.val$ph.0.10[t.ix],
  lower = t.lower, upper = t.upper
)

# add observed values on top 
points(
  1:nrow( d.ph10.val ),
  d.ph10.val$ph.0.10[t.ix], cex = 0.7,
  pch = c( 16, 1, 16)[t.col],
  col = c( "darkgreen", "black", "darkgreen" )[t.col]
)
points(ph.pred[t.ix], pch = 16, cex = 0.6, col = "grey60")

# Add meaningfull legend
legend( "topleft", 
        bty = "n", cex = 0.85,
        pch = c( NA, 16, 1, 16 ), pt.cex = 0.6, lwd = 1,
        lty = c( 1, NA, NA, NA ), 
        col = c( "grey", "grey60", "black", "darkgreen" ), 
        seg.len = 0.8,
        legend = c(
          "90 %-prediction interval", 
          paste0("prediction (n = ", nrow(d.ph10.val), ")"),
          paste0("observation within interval (n = ", 
                 sum( t.col %in% c(2) ), ")" ),
          paste0("observation outside interval (n = ", 
                 sum( t.col %in% c(1,3)), ", ", 
                 round(sum(t.col %in% c(1,3)) / 
                         nrow(d.ph10.val)*100,1), "%)") )
)


## ----create-coverage-probabilty-plots,fig.align='center', fig.pos = "h", fig.width=4,fig.height=4.5, out.width='0.45\\textwidth',fig.cap="Coverage probabilities of one-sided prediction intervals computed for the validation data set of topsoil pH of the Berne study area."----

# Coverage probabilities plot
# create sequence of nominal probabilities 
ss <- seq(0,1,0.01)
# compute coverage for sequence
t.prop.inside <- sapply(ss, function(ii){
  boot.quantile <-  t( apply(ph.pred.distribution, 1, quantile, 
                             probs = c(0,ii) ) )[,2]
  return( sum(boot.quantile <= d.ph10.val$ph.0.10)/nrow(d.ph10.val) )
})

plot(x = ss, y = t.prop.inside[length(ss):1], 
     type = "l", asp = 1,
     ylab = "coverage probabilities", 
     xlab="nominal probabilities" )
# add 1:1-line  
abline(0,1, lty = 2, col = "grey60")
# add lines of the two-sided 90 %-prediction interval
abline(v = c(0.05, 0.95), lty = "dotted", col = "grey20")


## ----session-info,results='asis'----------------------------------
toLatex(sessionInfo(), locale = FALSE)


## ----export-r-code,echo=FALSE,result="hide"-----------------------
purl("OpenGeoHub-machine-learning-training-2.Rnw")

