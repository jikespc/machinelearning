package com.yjx

import breeze.linalg.linspace
import breeze.plot._


object Test {
  def main(args: Array[String]): Unit = {
    val f = Figure()
    val p = f.subplot(0)
    val x = linspace(-100,100.0)
    **分别是函数2x,3x**
    p += plot(x,x:*2.0,'+')
    p += plot(x,x:*3.0, '.')
    p.xlabel = "x axis"
    p.ylabel = "y axis"
    //f.saveas("lines.png") // save current figure as a .png, eps and pdf also supported

    val p2 = f.subplot(2,1,1)
    val g = breeze.stats.distributions.Gaussian(0,1)
    p2 += hist(g.sample(100000),100)
    p2.title = "A normal distribution"
    //f.saveas("subplots.png")

    /*val f2 = Figure()
    f2.subplot(0) += image(DenseMatrix.rand(200,200))
    f2.saveas("image.png")*/
  }
}
