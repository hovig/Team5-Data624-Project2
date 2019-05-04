
#only_numeric <- bev %>% keep(is.numeric)

#only_numeric %>%
#  drop_na() %>%
#  cor() %>%
#  corrplot(order = 'hclust', addrect = 3, method = 'square')

#library(corrplot)
bev_raw <- read_csv('C://Users//pgood//OneDrive//Documents//DATA624//final//StudentData.csv') %>%
  drop_na(PH)


bev <- select(bev_raw, -PH)
cols <- str_replace(colnames(bev), '\\s', '')
colnames(bev) <- cols
bev <- mutate(bev, BrandCode = ifelse(is.na(BrandCode), 'Unknown', BrandCode))
y <- bev_raw$PH


zero_vars <- nearZeroVar(bev)
bev_new <- bev[, -zero_vars]

pred <- mice(data = bev_new, m = 5, method = 'rf', maxit = 3)
bev_imputed <- complete(pred)

form <- dummyVars(~ ., data = bev_imputed)
bev_imputed <- predict(form, bev_imputed) %>% data.frame() %>% as.matrix()
str(bev_imputed)

library(caretEnsemble)


xgb_grid <- expand.grid(eta = c(.025, .05), nrounds = c(750,1000), max_depth = c(3,5,7,9), gamma = c(0),
                          colsample_bytree = c(.6),
                          min_child_weight = c(.6,1),
                          subsample = c(.6))
cub_grid <- expand.grid(.committees = c(1,3,5), .neighbors = c(1,3,5,7,9))
svm_grid <- expand.grid(.C = c(.5,1,2,4,8,16,32,64,128), .sigma = c(.005,.01,.5, .1))
mars_grid <- expand.grid(.degree = c(1,2), .nprune = seq(16,36,4))
dart_grid <- expand.grid(eta = c(.025), nrounds = c(1000), gamma = c(.1), skip_drop = c(.2,.6), rate_drop = c(.2,.6), max_depth = c(5),
                          colsample_bytree = c(.6),
                          min_child_weight = c(.6),
                          subsample = c(.6))

tuning_list <-list(
  xgbt = caretModelSpec(method="xgbTree", tuneGrid = xgb_grid),
  xgbd = caretModelSpec(method="xgbDART", tuneGrid = dart_grid),
  cub = caretModelSpec(method="cubist", tuneGrid = cub_grid),
  svm = caretModelSpec(method="svmRadial", tuneGrid = svm_grid),
  mars = caretModelSpec(method="bagEarth", tuneGrid = mars_grid)
)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
set.seed(42)
my_control <- trainControl(method = 'cv',
                           number = 5,
                           savePredictions = 'final',
                           index = createFolds(y, 5),
                           allowParallel = TRUE) 

mod_list <- caretList(
  x = bev_imputed,
  y = y,
  preProcess = c('center', 'scale'),
  trControl = my_control,
  tuneList = tuning_list
)

show_results(mod_list$xgbt)
show_results(mod_list$xgbd)
show_results(mod_list$cub)
show_results(mod_list$svm)
show_results(mod_list$mars)

resamples <- resamples(mod_list)
modelCor(resamples)

ensemble1 <- caretStack(
  mod_list,
  method="glm",
  metric="RMSE",
  trControl=trainControl(
    method="cv",
    number=5,
    savePredictions="final"
  )
)
ensemble1

ensemble2 <- caretEnsemble(mod_list, 
                        metric = 'RMSE', 
                        trControl=trainControl(
                          method="cv",
                          number=5,
                          savePredictions="final"
                        ))

summary(ensemble2)

stopCluster(cl)