import tensorflow as tf
import os
from glob import glob
tf.version.VERSION
from tensorflow.keras.preprocessing import image_dataset_from_directory
print(tf.__version__)
import datasetOptimizer
numberOfClasses = 22
def train(product,sensor,new=True):
  train_dir = "/data/{}/{}/train/".format(product,sensor)
  classdirs = glob(f"{train_dir}/*")
  classes = []
  for c in classdirs:
    classes.append(c.split("/")[-1])


  validation_dir = "/data/{}/{}/validation/".format(product,sensor)
  BATCH_SIZE = 32
  IMG_SIZE = (224, 224)
  train_dataset = image_dataset_from_directory(train_dir,label_mode="categorical" ,labels="inferred", shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
  validation_dataset = image_dataset_from_directory(validation_dir,label_mode="categorical" ,labels="inferred", shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
  val_batches = tf.data.experimental.cardinality(validation_dataset)
  test_dataset = validation_dataset.take(val_batches // 5)
  validation_dataset = validation_dataset.skip(val_batches // 5)
  print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
  print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  
  # train_dataset = tf.keras.utils.to_categorical(train_dataset,numberOfClasses,dtype=float)
  # validation_dataset = tf.keras.utils.to_categorical(validation_dataset,numberOfClasses,dtype=float)
  train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
  validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

  test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
  layers = [
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
  ]
  data_augmentation = tf.keras.Sequential(layers)
  preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
  IMG_SHAPE = IMG_SIZE + (3,)
  modelPath ="/data/{}/{}/model/".format(product,sensor)
  checkpointPath ="/data/{}/{}/checkpoint/".format(product,sensor)
  if new:
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=True, weights='imagenet')
    base_model.trainable = False
    base_model.summary()
    image_batch, _ = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    # print(feature_batch.shape)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # feature_batch_average = global_average_layer(feature_batch)
    # print(feature_batch_average.shape)
    prediction_layer = tf.keras.layers.Dense(numberOfClasses,activation="softmax",name="logits")
    # prediction_batch = prediction_layer(feature_batch_average)
    # print(prediction_batch.shape)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=True)
    # x = global_average_layer(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    
    
    fine_tune_at = 100
    base_learning_rate = 0.00005
    # outputs = prediction_layer(x)

    outputs = tf.keras.layers.Activation('softmax')(tf.keras.layers.Dense(numberOfClasses)(x))
    model = tf.keras.Model(inputs, outputs)
    for layer in base_model.layers[:fine_tune_at]:# Freeze all the layers before the `fine_tune_at` layer
      layer.trainable =  False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
  else:
    model = tf.keras.models.load_model(modelPath)
    print("loaded model from {}".format(modelPath))
    for layer in model.layers:
      layer.trainable =True
 
  try: 
    os.rmdir(checkpointPath)
  except:
    print("no dir to delete this is not a problem") 
  try:
    os.mkdir(checkpointPath)
  except:
    print("checkpoint exists over writing!!!!")
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=2,)
  class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
      if(logs.get('val_accuracy') >= 1):
        self.model.stop_training = True
  myCallback = myCallback()
  fine_tune_at = 100
  
  
  model.summary()
  if new:
    initial_epochs = 100
  else:
    initial_epochs = 5
  # loss0, accuracy0 = model.evaluate(test_dataset)
  # print("initial loss: {:.8f}".format(loss0))
  # print("initial accuracy: {:.8f}".format(accuracy0))
  
  
  history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset,callbacks=[model_checkpoint_callback, myCallback])
  if new:
    base_model.trainable = True
     # Fine-tune from this layer onwards
    for layer in base_model.layers[:fine_tune_at]:# unFreeze all the layers to fine tune.
      layer.trainable =  True
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), metrics=['accuracy'])
    fine_tune_epochs = 150
    UnfrozzenEpochs  = 150
    total_epochs =  initial_epochs + UnfrozzenEpochs
    history = model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset,callbacks=[model_checkpoint_callback])
    _, accuracy = model.evaluate(test_dataset)

    total_epochs =  initial_epochs + UnfrozzenEpochs +fine_tune_epochs
    # for layer in base_model.layers[:fine_tune_at]:# Freeze all the layers before the `fine_tune_at` layer
    #   layer.trainable =  False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
  
    history = model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset,callbacks=[model_checkpoint_callback])
  
  # model = tf.keras.models.load_model(checkpointPath) 
  _, accuracy = model.evaluate(test_dataset)
  print('Test accuracy :', accuracy)    
  print("the best version of the model was saved to /data/{}/{}/model/".format(product,sensor))
  model.save("/data/{}/{}/test/".format(product,sensor))
  # model.save("/data/{}/{}/model/".format(product,sensor))
  # model.save("/data/{}/{}/model.h5",save_format = "h5")
  # tf.keras.models.save_model(model,"/data/{}/{}/save_model")
  testModel = tf.keras.models.load_model("/data/{}/{}/test/".format(product,sensor))
  _, accuracy = testModel.evaluate(test_dataset)
  print('Test accuracy :', accuracy)
  model.save(modelPath)    
  pred = model.predict(train_dataset)
  print(f"pred = {pred} shape= {pred.shape}")
  print("the best version of the model was saved to {}".format(modelPath))
  return accuracy


if __name__ == "__main__":
  product = "polyclass"
  sensor = "camera"
  print(f"the score was = {train(product,sensor,new=True)}")
  # sensor = "laser"
  print(f"the score was = {train(product,sensor,new=False)}")
  datasetOptimizer.polyClassErrorSearch()
  # sensor = "camera"
  # # print(f"the score was = {train(product,sensor,new=False)}")
  # sensor = "laser"
  # print(f"the score was = {train(product,sensor,new=False)}")
  # datasetOptimizer(product)