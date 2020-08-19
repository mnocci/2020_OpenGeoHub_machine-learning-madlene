# code translation from Madlene Nussbaum workshop presentation to a tidymodels format
#prepared by John Lewis June 2,2020
library(tidyverse)
library(tidymodels)
library(geoGAM)
data(berne)

class(berne$dclass) <- "factor"

br_df <- berne %>%
     filter(dataset=='calibration') %>%
     select(-c(site_id_unique,dataset, waterlog.30,waterlog.50,waterlog.100,
             ph.0.10,ph.10.30,ph.30.50,ph.50.100)) %>%
     drop_na()
glimpse(br_df)

set.seed(42) # makes sample() reproducible
#code for generating 30 predictor sample variables
l_covar <- names(br_df[, 4:ncol(br_df)])
br_fin <- br_df[1:620 , l.covar[sample(1:length(l_covar), 30)]]
dclass <- br_df[, 3]
br_f <- cbind(br_fin, dclass)
br_f <- as_tibble(br_f)
br_fn <- br_f %>%
   select(dclass, everything())

glimpse(br_fn)

#preprocessing - feature engineering
br_rec <- recipe(dclass ~ ., data = br_fn) %>%
  step_dummy(all_nominal(),-all_outcomes()) %>% # for categoricals in sample
  step_zv(all_numeric(), -all_outcomes()) %>% # exclude variables with zero var.
  step_center(all_numeric(), -all_outcomes()) %>% #normalize numeric data
  step_scale(all_numeric(), -all_outcomes())



#model spcification
br_spec <- multinom_reg(penalty= 2, mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

#workflow
wf <- workflow() %>%
  add_recipe(br_rec)

#fitting train model
br_fit <- wf %>%
  add_model(br_spec) %>%
  fit(data = br_fn)

br_fit %>%
  pull_workflow_fit() %>%
  tidy()


#Tune lasso parameters
#since there were only ~ 30 variables I chose to use bootstrapping rather than cv
set.seed(1234)
br_boot <- bootstraps(br_fn)

#penalty = lambda & mixture = alpha for glmnet
tune_spec <- multinom_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

#tuning grid
lambda_grid <- grid_regular(penalty(), levels = 30)

#Now it’s time to tune the grid, using the workflow object.

doParallel::registerDoParallel()

set.seed(2020)
br_grid <- tune_grid(wf %>%
  add_model(tune_spec),
  resamples = br_boot,
  grid = lambda_grid
)

#collecting the metrics
br_grid %>%
  collect_metrics()

#visualize results based on accuracy and roc_auc
br_grid %>%
  collect_metrics() %>%
  ggplot(aes(penalty, mean, color = .metric)) +
  geom_errorbar(aes(
    ymin = mean - std_err,
    ymax = mean + std_err
  ),
  alpha = 0.5
  ) +
  geom_line(size = 1.5) +
  facet_wrap(~.metric, scales = "free", nrow = 2) +
  scale_x_log10() +
  theme(legend.position = "none")

# I chose accuracy in fitting the final trained model
best_acc <- br_grid %>%
  select_best("accuracy")

final_br <- finalize_workflow(
  wf %>% add_model(tune_spec),
  best_acc
)

#visualize varaiable importance
library(vip)
final_br %>%
  fit(br_fn) %>%
  pull_workflow_fit() %>%
  vi(lambda = best_acc$penalty) %>%
  mutate(
    Importance = abs(Importance),
    Variable = fct_reorder(Variable, Importance)
  ) %>%
  ggplot(aes(x = Importance, y = Variable, fill = Sign)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(y = NULL)


########################################
#Random Forest
# each tuning steps took ~30 minutes on my 2 core machine
# I ran it on the complete set of predictor variables
br_df <- berne %>%
     filter(dataset=='calibration') %>%
     select(-c(site_id_unique, dataset, dclass,waterlog.30,waterlog.50,waterlog.100,
            ph.10.30,ph.30.50,ph.50.100)) %>%
     drop_na()
glimpse(br_df)

br_df %>%
  ggplot(aes(x, y, color = ph.0.10)) +
  geom_point(size = 0.75, alpha = 0.4) +
  ggtitle("Berne pH 0-10cm")+
  xlab("easting") +
  ylab("northing")
  labs(color = NULL)

brtest_df <- berne %>%
     filter(dataset=='validation') %>%
     select(-c(site_id_unique, dataset, dclass,waterlog.30,waterlog.50,waterlog.100,
            ph.10.30,ph.30.50,ph.50.100)) %>%
     drop_na()
glimpse(brtest_df)

br_f <- as_tibble(br_df)
br_fn <- br_f %>%
   select(-c(x,y)) %>%
   select(ph.0.10, everything())

glimpse(br_fn)

#recipe - preprocessing
br_rec <- recipe(ph.0.10 ~ ., data = br_fn) %>%
  step_dummy(all_nominal(),-all_outcomes()) %>% # for categorical in sample
  step_zv(all_numeric(), -all_outcomes())  # exclude variables with zero var.

#prepping & juicing
br_pr <- prep(br_rec)
juice(br_pr) #just for information now

#model specification
tune_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_mode("regression") %>%
  set_engine("ranger")

#mtry & min_n are hyperparameters that can’t be learned from data when
#training the model

#create workflow object
tune_wf <- workflow() %>%
  add_recipe(br_rec) %>%
  add_model(tune_spec)

#train hyperparameters
set.seed(234)
br_folds <- vfold_cv(br_fn)

doParallel::registerDoParallel()

#tune - first try with simple grid-choose 20 grid points automatically
set.seed(345)
tune_res <- tune_grid(
  tune_wf,
  resamples = br_folds,
  grid = 20
)

tune_res

#visualize results of tuning
tune_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
    values_to = "value",
    names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "rmse")

#set up second grid
rf_grid <- grid_regular(
  mtry(range = c(50, 150)),
  min_n(range = c(2, 8)),
  levels = 5
)

rf_grid

#tune one more time, but this time in a more targeted way with the rf_grid.
set.seed(456)
regular_res <- tune_grid(
  tune_wf,
  resamples = br_folds,
  grid = rf_grid
)

regular_res

#visualize these results
regular_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "rmse")

# use best tune hyperparametrs to fit final model
best_rmse <- select_best(regular_res, "rmse")

final_rf <- finalize_model(
  tune_spec,
  best_rmse
)

final_rf

library(vip)
#final model - variables of importance
final_rf %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(ph.0.10 ~ .,
    data = juice(br_pr)
  ) %>%
  vip(geom = "point")


######predict test data -- I didn't run this since the data was already split
#so if I ran it in tidymodels the data splitting would possible be different and not
#replicate your split results. But this could be done easily.


final_wf <- workflow() %>%
  add_recipe(br_rec) %>%
  add_model(final_rf)

final_res <- final_wf %>%
  last_fit(br_split)

#metrics of predict results
final_res %>%
  collect_metrics()

