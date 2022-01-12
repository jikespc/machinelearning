package com.yjx

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.{Row, SparkSession}

object Word2Vec {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().master("local").appName("WordToVec").getOrCreate()
    //输入数据
    val documentDF = sparkSession.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(8)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)

    val result = model.transform(documentDF)
    result.collect().foreach {
      case Row(text: Seq[_], features) => println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n")
    }
  }
}
