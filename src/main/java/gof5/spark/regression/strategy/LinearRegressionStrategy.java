package gof5.spark.regression.strategy;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

public class LinearRegressionStrategy extends Template implements Strategy {

	private static final org.slf4j.Logger log = LoggerFactory.getLogger(LinearRegressionStrategy.class);
	private LinearRegressionModel sameModel;

	@Override
	public void execute(JavaSparkContext sc, JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData,
			String modelPath) {
		// Model Creation
		int numIterations = 100;
		double stepSize = 0.00000001;
		LinearRegressionModel model = LinearRegressionWithSGD.train(trainingData.rdd(), numIterations, stepSize);
		// Model Evaluation
		JavaPairRDD<Object, Object> valuesAndPreds = testData
				.mapToPair(point -> new Tuple2<>(model.predict(point.features()), point.label()));
		super.ModelEvaluate(valuesAndPreds);
		// Model Save & Load
		super.outputPrepare(modelPath);
		model.save(sc.sc(), modelPath);
		sameModel = LinearRegressionModel.load(sc.sc(), modelPath);
	}

	@Override
	public double predict(double[] param) {
		Vector newData = Vectors.dense(param);
		double prediction = sameModel.predict(newData);
		log.info("Model Prediction on New Data = " + prediction);
		return prediction;
	}

}
