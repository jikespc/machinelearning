package com.yjx

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object Number {
  def main(args: Array[String]): Unit = {
    val session = SparkSession.builder().master("local").appName("Number").getOrCreate()
    //读取数据
    val dateFrame = session.read.format("libsvm").load("src/main/data/sample_number.txt")
    //将数据分为训练数据和测试数据
    val Array(trainingData,testData) = dateFrame.randomSplit(Array(0.7, 0.3))
    //创建线性回归
    val linearRegression = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFitIntercept(true)
    //开始适配模型
    val linearRegressionModel = linearRegression.fit(trainingData)
    println("线性回归系数：" + linearRegressionModel.coefficients)
    println("线性回归截断值：" + linearRegressionModel.intercept)

    //获取模型的综合评估信息
    val summary = linearRegressionModel.summary
    println("根据训练用例" + summary.rootMeanSquaredError)

    //预估值
    println("计算向量(1,2,3)的预估值" + linearRegressionModel.predict(Vectors.dense(1, 2, 3)))

    //根据测试数据获取模型的评估值
    val regressionSummary = linearRegressionModel.evaluate(testData)
    println("根据测试用例获取：" + regressionSummary.rootMeanSquaredError)

    //保存模型
    //linearRegressionModel.save("models/number_" + UUID.randomUUID())

  }
}
