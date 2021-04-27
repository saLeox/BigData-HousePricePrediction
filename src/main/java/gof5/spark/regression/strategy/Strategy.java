package gof5.spark.regression.strategy;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;

public interface Strategy {

	public void execute(JavaSparkContext sc, JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData,
			String modelPath);

	public double predict(double[] param);

}
