import tensorflow as tf
import numpy as np
import cv2
import random

# 이미지 확인하기 위한 라이브러리 선언.
from matplotlib import pyplot
from scipy.misc import toimage

# cifar10 데이터셋을 얻기위한 데이터셋 라이브러리
from keras.datasets import cifar10


########################################################################
# Model Class
class Model:
    def __init__(self, sess, name, c_image_size_flat, c_num_class, c_lr):
        self.sess = sess
        self.name = name
        self.learning_rate = c_lr
        self.img_size, self.num_class = c_image_size_flat, c_num_class
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, self.img_size])
            self.Y = tf.placeholder(tf.float32, [None, self.num_class])
            self.keep_prob = tf.placeholder(tf.float32)

            # Layer 1
            w1 = tf.get_variable("W1", shape=[self.img_size, 1024],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([1024]))
            L1 = tf.nn.relu(tf.matmul(self.X, w1) + b1)
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            # Layer 2
            w2 = tf.get_variable("W2", shape=[1024, 1024],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([1024]))
            L2 = tf.nn.relu(tf.matmul(L1, w2) + b2)
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            # Layer 3
            w3 = tf.get_variable("W3", shape=[1024, 1024],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([1024]))
            L3 = tf.nn.relu(tf.matmul(L2, w3) + b3)
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

            # Layer 4
            w4 = tf.get_variable("W4", shape=[1024, 1024],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([1024]))
            L4 = tf.nn.relu(tf.matmul(L3, w4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            # Layer 5
            w5 = tf.get_variable("W5", shape=[1024, self.num_class],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([self.num_class]))
            self.hypothesis = tf.matmul(L4, w5) + b5

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.hypothesis, labels=self.Y))
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test, self.keep_prob: 1})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: 0.85})


# 함수 영역
def line():
    print("----------------------------------------------------------")


def image_show(img_list):
    # 이미지 확인용 함수
    # 3 x 3의 subplot 에 이미지를 출력한다. (9개)
    for img_i in range(0, 9):
        pyplot.subplot(330 + 1 + img_i)
        pyplot.imshow(toimage(img_list[img_i]))
    # 이미지를 출력한다.
    pyplot.show()


def next_batch(img_data, label_data, batch_size_, index_):
    batch_img_data = img_data[batch_size_ * index_: batch_size_ * (index_ + 1)]
    batch_label_data = label_data[batch_size_ * index_: batch_size_ * (index_ + 1)]
    return batch_img_data, batch_label_data


########################################################################
image_show_sw = False
# keras dataset 에서 cifar10 데이터셋을 얻는다.
(image_train, class_train), (image_test, class_test) = cifar10.load_data()
data_class_info = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# print(data_class_info)

# 이미지를 확인하기 위한 함수
if image_show_sw:
    image_show(image_train)
########################################################################

# 이미지에 대한 정보
# 이미지의 사이즈, 채널의 개수(R,G,B)
img_size, img_channels = 32, 3
image_size_flat = img_size * img_size * img_channels
# or image_size_flat = x_train[0].size 으로 쓸 수도 있다.
# 클래스의 개수 (분류하는 라벨의 개수)
num_classes = 10
# 학습시켜야하는 이미지의 개수
num_images_train = int(image_train.size / image_size_flat)
num_images_test = int(image_test.size / image_size_flat)

# # Blur Test
# for i in range(num_images_train):
#     image_train[i] = cv2.GaussianBlur(image_train[i], (3, 3), 0)
# for i in range(num_images_test):
#     image_test[i] = cv2.GaussianBlur(image_test[i], (3, 3), 0)

# Image flat & Normalization
flat_image_train = image_train.reshape(-1, image_size_flat) / 255.
flat_image_test = image_test.reshape(-1, image_size_flat) / 255.
# print(flat_image_train[0])

# one-hot encoding
labels_train_tmp = tf.reshape(tf.one_hot(class_train, depth=num_classes), shape=[-1, num_classes])
labels_test_tmp = tf.reshape(tf.one_hot(class_test, depth=num_classes), shape=[-1, num_classes])

# 입력 확인용
sess = tf.Session()
one_hot_labels_train = sess.run(labels_train_tmp)
one_hot_labels_test = sess.run(labels_test_tmp)

# gray_img_train, gray_img_test = [], []
# for img in image_train:
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     gray_img_train.append(img)
# for img in image_test:
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     gray_img_test.append(img)


# 이미지 정보 출력
line()
print("[이미지 정보]")
print("이미지의 크기 : {} x {} , 채널의 개수 : {}".format(img_size, img_size, img_channels))
print("학습 이미지의 개수 :", num_images_train)
print("테스트 이미지의 개수 :", num_images_test)
line()

# 이미지 라벨링 확인용
# for inn in range(9):
#     print(data_class_info[int(class_test[inn])])
# print(one_hot_labels_test[:9])
# image_show(image_test)

########################################################################
# 학습 정보
learning_rate = [0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005]
training_epochs = 1000
batch_size = 2000

# initialize
# sess = tf.Session()
models = []
num_models = 10
for m in range(num_models):
    models.append(Model(sess, "model" + str(m), image_size_flat, num_classes, learning_rate[m]))

sess.run(tf.global_variables_initializer())
print("[학습 시작]")
########################################################################
# Training CIFAR-10 data.
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(num_images_train / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(flat_image_train, one_hot_labels_train, batch_size, i)

        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'Cost =', avg_cost_list)

print('[학습 종료]')
########################################################################
# Test model check accuracy
# Test Accuracy
print('\n[Test Accuracy]')
predictions = np.zeros([num_images_test, 10])
for m_idx, m in enumerate(models):
    print("Test model", m_idx, "Accuracy: ", m.get_accuracy(
        flat_image_test, one_hot_labels_test))
    p = m.predict(flat_image_test)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.arg_max(predictions, 1),
                                       tf.arg_max(one_hot_labels_test, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

print('Test data Accuracy :', float(sess.run(ensemble_accuracy)) * 100, '%')

########################################################################


########################################################################


########################################################################