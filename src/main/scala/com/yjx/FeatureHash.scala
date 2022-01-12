package com.yjx

import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.sql.SparkSession

object FeatureHash {
  def main(args: Array[String]): Unit = {
    //创建环境
    val sparkSession = SparkSession.builder().master("local").appName("Hello06NaiveBayes").getOrCreate()
    //转换数据
    val dataset = sparkSession.createDataFrame(Seq(
      (2.2, true, "1", "foo"),
      (3.3, false, "2", "bar"),
      (4.4, false, "3", "baz"),
      (5.5, false, "4", "foo")
    )).toDF("real", "bool", "stringNum", "string")

    val hasher = new FeatureHasher()
      .setInputCols("real", "bool", "stringNum", "string")
      .setOutputCol("features")

    val featurized = hasher.transform(dataset)
    featurized.show(false)
  }
}
