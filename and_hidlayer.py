# -*- coding: utf-8 -*-
## @package and_hidlayer.py
# and_hidlayer.py  And operation for neural-network test program
# Copyright (c) 2017 Syunsuke Okamoto
#
# This software is released under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# 

#tensorflow のインポート
import tensorflow as tf
import time

# タイマスタート
start = time.time()

## 式(グラフ)の作成

# 入力層 x　(2Unit)
x = tf.placeholder(tf.float32, [None, 2])
# 重み付け W
W = tf.Variable(tf.random_uniform([2,3], -1.0, 1.0))
# しきい値 b
b = tf.Variable(tf.random_uniform([3], -1.0, 1.0))
# 中間層 y (シグモイド関数使用) 行列演算 ς(x・W +b) もしくは  ς{Σ(xiWi + bi)}
y = tf.nn.sigmoid(tf.matmul(x, W) + b)

# 重み付け W2
W2 = tf.Variable(tf.random_uniform([3,1], -1.0, 1.0))
# しきい値 b2
b2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 出力層 y2 (シグモイド関数使用) 行列演算 ς(y・W2 +b2) もしくは  ς{Σ(yiW2i + b2i)}
y2 = tf.nn.sigmoid(tf.matmul(y, W2) + b2)

# 教師信号
y_ = tf.placeholder(tf.float32, [None, 1])

# 線形回帰により lossを演算（教師信号と現在の出力値との誤差から演算)
loss = tf.reduce_mean(tf.square(y2 - y_))
# 勾配降下法によるオプティマイザー (重み 0.1)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# 線形回帰により loss2を演算（教師信号と現在の出力値との誤差から演算)
loss2 = tf.reduce_mean(tf.square(y - loss))
# 勾配降下法によるオプティマイザー (重み 0.1)
train_step2 = tf.train.GradientDescentOptimizer(0.5).minimize(loss2)

#保存設定
saver = tf.train.Saver()

# 変数初期化
init = tf.initialize_all_variables()

# セッション作成
sess = tf.Session()

# 変数初期化セッション実行
sess.run(init)
# 教師信号 入力値x_1 (2入力論理積パターン)
x_1=[[0,0],[0,1],[1,0],[1,1]]
# 教師信号 出力値y_1 (2入力論理積パターン)
y_1=[[0],[0],[0],[1]]

# テスト信号 入力値 x_test 
x_test=[[0.8,0.8]]

# 学習回数 10000
for i in range(10000):
        # 学習の実行(train_step) xにx_1, yにy_1を代入して実行
    sess.run(train_step,feed_dict={x: x_1,y_:y_1})
         # 学習の実行(train_step2) xにx_1, yにy_1を代入して実行
    sess.run(train_step2,feed_dict={x: x_1,y_:y_1})   
        # もし iが 100で割りきれるならば、xにx_1を代入して実行し yについて出力
    if i % 100 == 0:
        print i,sess.run(y2,feed_dict={x: x_1})

# for文終わり   
# xにx_testを代入して実行し yについて出力(学習パターン以外のデータをいれたときの判定）    
print sess.run(y2,feed_dict={x: x_test}) 

#ファイル保存
saver.save(sess, "and_hidlayer_10000.ckpt")
#セッション終了
sess.close()


# 経過時間取得
timer = time.time() - start
# 経過時間の表示
print ("time:{0}".format(timer)) + "[sec]"
