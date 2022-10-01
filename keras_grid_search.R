load('split_data.Rdata')

FLAGS <- flags(flag_numeric("nodes_layer1", 256),
               flag_numeric("nodes_layer2", 128),
               flag_numeric("nodes_layer3", 64),
               flag_numeric("dropout_layer1", 0.2),
               flag_numeric("dropout_layer2", 0.2),
               flag_numeric("dropout_layer3", 0.2))

model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes_layer1, activation = "relu", input_shape = ncol(x)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout_layer1) %>%
  layer_dense(units = FLAGS$nodes_layer2, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout_layer2) %>%
  layer_dense(units = FLAGS$nodes_layer3, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout_layer3) %>%
  layer_dense(units = 2, activation = "sigmoid") %>%
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_rmsprop(), 
          metrics = "roc") %>% 
  fit(x = x, 
      y = y, 
      epochs = 50, 
      batch_size = 256,
      validation_split = 0.2,
      callbacks = list(callback_early_stopping(patience = 20),
                       callback_reduce_lr_on_plateau()),
      verbose = 2) 

