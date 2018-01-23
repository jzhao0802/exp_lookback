library(tidyverse)
library(feather)
setwd("F:/Valentina/Experiments/exp_lookback/")
filename <- "F:/Projects/Strongbridge/data/modelling/Advanced_model_data/05_combined_train_unmatched_test_capped_freq_datediff.rds"
df <- readRDS(filename)
df <- as.data.frame(df)
write_feather(df, "./extra_data/5_combined_train_unmatched_test_capped_freq_datediff.feather")
df_2 <- read_feather("./extra_data/5_combined_train_unmatched_test_capped_freq_datediff.feather")
