package com.yjx

import org.apache.spark.sql.SparkSession

object DataLibSvm {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().appName("DataLibSvm").master("local").getOrCreate()

    val value = sparkSession.read.format("libsvm").textFile("src/main/data/健康状况训练集.txt")
    value.show()
    value.printSchema()
  }
}
