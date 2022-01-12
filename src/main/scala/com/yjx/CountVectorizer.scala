package com.yjx

import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.SparkSession

object CountVectorizer {
  def main(args: Array[String]): Unit = {
    val session = SparkSession.builder().master("local").appName("CountVectorizer").getOrCreate()

    val dataFrame = session.createDataFrame(Seq(
      (0, Array("a", "e", "b", "c", "e")),
      (1, Array("a", "b", "b", "c", "a")),
      (2, Array("a", "c")),
      (3, Array("a", "b", "c", "d", "e", "f", "e", "e", "d")),
      (4, Array("a", "b", "c", "d", "e", "f", "g", "h"))
    )).toDF("id", "words")

    val vectorizerModel = new CountVectorizer().setOutputCol("features").setInputCol("words").fit(dataFrame)
    vectorizerModel.transform(dataFrame).show(false)
    println(vectorizerModel.vocabulary.mkString(","))
  }
}
