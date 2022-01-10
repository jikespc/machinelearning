package com.yjx

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{Dataset, SparkSession}

/**
 * 模型评估指标是指测试集的评估指标，而不是训练集的评估指标
 * "rmse" (default): root mean squared error
 * "mse": mean squared error
 * "r2": R2 metric
 * "mae": mean absolute error
 */
object LinearRegression {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().master("local").appName("LinearRegression").getOrCreate()
    //读取数据
    val value = sparkSession.read.format("libsvm").textFile("src/main/data/lpsa.data")
    import sparkSession.implicits._
    //转换成labeledPoint
    val trainingData: Dataset[LabeledPoint] = value.map(ele => {
      val label = ele.split(",")(0).toDouble
      val array = ele.split(",")(1).split("\\s").map(_.toDouble)
        LabeledPoint(label, Vectors.dense(array))
    })

    //进行线性回归的建模
    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFitIntercept(true)


    val lrModel: LinearRegressionModel = lr.fit(trainingData)

    //查看模型的信息
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

  }
}
