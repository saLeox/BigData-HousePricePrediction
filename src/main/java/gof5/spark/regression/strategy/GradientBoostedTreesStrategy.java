package gof5.spark.regression.strategy;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

public class GradientBoostedTreesStrategy extends Template implements Strategy {
	private static final org.slf4j.Logger log = LoggerFactory.getLogger(GradientBoostedTreesStrategy.class);

	private GradientBoostedTreesModel sameModel;

	@Override
	public void execute(JavaSparkContext sc, JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData,
			String modelPath) {
		// Train a GradientBoostedTrees model.
		// The defaultParams for Regression use SquaredError by default.
		BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Regression");
		boostingStrategy.setNumIterations(3); // Note: Use more iterations in practice.
		boostingStrategy.getTreeStrategy().setMaxDepth(5);
		// Empty categoricalFeaturesInfo indicates all features are continuous.
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);

		GradientBoostedTreesModel model = GradientBoostedTrees.train(trainingData, boostingStrategy);

		// Evaluate model on test instances and compute test error
		JavaPairRDD<Object, Object> valuesAndPreds = testData
				.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		super.ModelEvaluate(valuesAndPreds);
		// Model Save & Load
		super.outputPrepare(modelPath);
		model.save(sc.sc(), modelPath);
		sameModel = GradientBoostedTreesModel.load(sc.sc(), modelPath);
	}

	@Override
	public double predict(double[] param) {
		Vector newData = Vectors.dense(param);
		double prediction = sameModel.predict(newData);
		log.info("Model Prediction on New Data = " + prediction);
		return prediction;
	}

}
