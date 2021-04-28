package gof5.spark.regression.strategy;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

public class DecisionTreeStrategy extends Template implements Strategy {
	private static final org.slf4j.Logger log = LoggerFactory.getLogger(DecisionTreeStrategy.class);

	private DecisionTreeModel sameModel;

	@Override
	public void execute(JavaSparkContext sc, JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData,
			String modelPath) {
		// Model Creation
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		String impurity = "variance";
		int maxDepth = 5;
		int maxBins = 32;
		// Train a DecisionTree model.
		DecisionTreeModel model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
				maxDepth, maxBins);
		// Model Evaluation
		JavaPairRDD<Object, Object> valuesAndPreds = testData
				.mapToPair(point -> new Tuple2<>(model.predict(point.features()), point.label()));
		super.ModelEvaluate(valuesAndPreds);
		// Model Save & Load
		super.outputPrepare(modelPath);
		model.save(sc.sc(), modelPath);
		sameModel = DecisionTreeModel.load(sc.sc(), modelPath);

	}

	@Override
	public double predict(double[] param) {
		Vector newData = Vectors.dense(param);
		double prediction = sameModel.predict(newData);
		log.info("Model Prediction on New Data = " + prediction);
		return prediction;
	}
}
