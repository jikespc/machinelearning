package com.yjx

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
 * 创建向量和标记点
 */
object DataLabeledPoint {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().appName("DataLabeledPoint").master("local").getOrCreate()
    //稠密向量
    val vector = Vectors.dense(1, 2)
    //稀疏向量
    val vector1 = Vectors.sparse(5, Array(2), Array(1))
    val labeledPoint = LabeledPoint(3, vector)

    println(labeledPoint)
//    println(vector)
//    println(vector1.toDense)
  }
}
