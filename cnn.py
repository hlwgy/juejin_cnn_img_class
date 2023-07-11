# %% 环境准备 =======================================================
# 导入包
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import numpy as np
# 强制禁用GPU，使用CPU。初学者安装GPU支持比较复杂，关键词：CUDA、cuDNN。
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'
# %% 数据准备 =======================================================
data_dir = "datasets" # 训练数据文件夹
epochs = 10 # 训练轮数
batch_size = 24 # 一个批次的数据量
img_height = 200 # 输入图片高度
img_width = 200 # 输入图片宽度
names = ['cloudy', 'rain', 'shine'] # 分类名称
model_path = 'tf/checkpoint'  # 训练结果存放的位置

# 训练数据
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# 验证数据
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# 增加缓存机制。小数据量无所谓，当大数据时可以提高效率
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %% 模型准备 =======================================================
# 创建模型，采用方法可以实现共用
def create_model():
  model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3)
  ])
  # 配置模型
  model.compile(optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy'])
  # 如果有权重，加载权重
  if os.path.exists(model_path + '.index'):
    model.load_weights(model_path)  

  return model
# %% 训练数据 =======================================================
def train():
  model = create_model()
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, 
                                            save_weights_only=True,
                                            save_best_only=True)
  history = model.fit(train_ds,validation_data=val_ds,epochs=epochs, callbacks=[cp_callback])
  
  # 可视化训练结果
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

# 预测数据 =======================================================
def predict(img_path):
  # 加载模型
  model = create_model()
  # 加载图片
  img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  # 开始预测
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  print( "分类 {}, 得分 {:.2f}".format(names[np.argmax(score)], 100*np.max(score)))

#%% 主方法。自行屏蔽问题
if __name__ == "__main__":
  # 训练
  train()
  # 预测演示
  # predict("test.png")
