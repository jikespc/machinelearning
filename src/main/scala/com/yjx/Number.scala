package com.yjx

import org.apache.spark.sql.SparkSession

object Number {
  def main(args: Array[String]): Unit = {
    val session = SparkSession.builder().master("local").appName("Number").getOrCreate()
    //读取数据
    val dateFrame = session.read.format("libsvm").load("src/main/data/sample_number.txt")
    //将数据分为训练数据和测试数据
    val Array(trainingData,testData) = dateFrame.randomSplit(Array(0.7, 0.3))
    //创建线性回归

  }
}
