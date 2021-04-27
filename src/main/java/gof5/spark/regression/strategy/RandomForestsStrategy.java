package gof5.spark.regression.strategy;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

public class RandomForestsStrategy extends Template implements Strategy {
	private static final org.slf4j.Logger log = LoggerFactory.getLogger(RandomForestsStrategy.class);

	private RandomForestModel sameModel;

	@Override
	public void execute(JavaSparkContext sc, JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData,
			String modelPath) {
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		int numTrees = 3; // Use more in practice.
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "variance";
		int maxDepth = 4;
		int maxBins = 32;
		int seed = 12345;
		// Train a RandomForest model.
		RandomForestModel model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo, numTrees,
				featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
		// Model Evaluation
		JavaPairRDD<Object, Object> valuesAndPreds = testData
				.mapToPair(point -> new Tuple2<>(model.predict(point.features()), point.label()));
		super.ModelEvaluate(valuesAndPreds);
		// Model Save & Load
		super.outputPrepare(modelPath);
		model.save(sc.sc(), modelPath);
		sameModel = RandomForestModel.load(sc.sc(), modelPath);
	}

	@Override
	public double predict(double[] param) {
		Vector newData = Vectors.dense(param);
		double prediction = sameModel.predict(newData);
		log.info("Model Prediction on New Data = " + prediction);
		return prediction;
	}

}
