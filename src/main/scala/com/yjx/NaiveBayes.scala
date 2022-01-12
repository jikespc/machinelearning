package com.yjx

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.types.{ArrayType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

/**
 * 朴素贝叶斯
 */
object NaiveBayes {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().master("local").appName("NaiveBayes").getOrCreate()
    //读取数据
    val lineRdd = sparkSession.read.textFile("src/main/data/sample_string.txt").rdd
    //转成labeledPoint
    val dataRdd = lineRdd.map(line => {
      var label = 0
      if ("yes".equals(line.split(",")(0))) {
        label = 1
      }
      var words = line.split(",")(1).split("\t").filter(_.length > 0)
      Row(label, words)
    })
    //构建类型
    val schema = StructType(List(
      StructField("label",IntegerType,nullable = false),
      StructField("words",ArrayType(StringType,true),nullable = false)
    ))
    //转成DataFrame
    val lineFrame = sparkSession.createDataFrame(dataRdd, schema)

    //转换将array<String>转成array<double>
    val countVectorizer = new CountVectorizer()
    countVectorizer.setInputCol("words").setOutputCol("features")
    val vectorizerModel = countVectorizer.fit(lineFrame)
    val dataFrame = vectorizerModel.transform(lineFrame)

    //拆分数据
    val Array(trainingData,testData) = dataFrame.randomSplit(Array(0.5, 0.5))

    //开始进行计算
    val model = new NaiveBayes().fit(trainingData)
    //转换数据
    val predictions = model.transform(testData)
    predictions.foreach(e=>println(e))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")

    sparkSession.stop()
  }
}
