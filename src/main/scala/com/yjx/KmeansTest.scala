package com.yjx

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession

object KmeansTest {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().master("local").appName("Kmeans").getOrCreate()
    import sparkSession.implicits._
    //读取数据
    val kmFrame = sparkSession.read.textFile("src/main/data/sample_km.txt").map(_.split("\t")).toDF("hero")
    //转成向量
    val word2Vec = new Word2Vec().setInputCol("hero").setOutputCol("features").setNumPartitions(4).setVectorSize(2)
    val word2VecModel = word2Vec.fit(kmFrame)
    val dataFrame = word2VecModel.transform(kmFrame)
    //开始计算
    val kMeans = new KMeans().setK(4).setSeed(1L)
    val kMeansModel = kMeans.fit(dataFrame)
    kMeansModel.clusterCenters.foreach(e=>println("Cluster Centers:" + e))

    val predictions = kMeansModel.transform(dataFrame)
    predictions.collect().sortBy(_.get(2).toString).foreach(line=>println("predictions:[原始数据]" + line.get(0) + "[向量数据]" + line.get(1) + "[所属分区]" + line.get(2)))

    //通过计算轮廓分数来评估聚类
    val evaluator = new ClusteringEvaluator()
    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")
  }
}
