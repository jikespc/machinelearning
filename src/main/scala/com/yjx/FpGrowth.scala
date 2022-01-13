package com.yjx

import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.SparkSession

object FpGrowth {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().master("local").appName("FpGrowth").getOrCreate()
    import sparkSession.implicits._
    val dataFrame = sparkSession.read.textFile("src/main/data/shopping_cart").map(_.split(",")).toDF("goods")

    val fPGrowth = new FPGrowth().setItemsCol("goods").setMinSupport(0.2).setMinConfidence(0.2)
    val growthModel = fPGrowth.fit(dataFrame)
    
    growthModel.freqItemsets.show()
    growthModel.associationRules.show()
    growthModel.transform(dataFrame).show()
  }
}
