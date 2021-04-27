package gof5.spark.regression.strategy;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.slf4j.LoggerFactory;

public abstract class Template {
	private static final org.slf4j.Logger log = LoggerFactory.getLogger(Template.class);

	// refer to https://spark.apache.org/docs/1.6.3/mllib-evaluation-metrics.html
	public void ModelEvaluate(JavaPairRDD<Object, Object> rdd) {
		RegressionMetrics metrics = new RegressionMetrics(rdd.rdd());
		// Squared error
		log.info("MSE = {}", metrics.meanSquaredError());
		log.info("RMSE = {}", metrics.rootMeanSquaredError());
		// R-squared
		log.info("R Squared = {}", metrics.r2());
		// Mean absolute error
		log.info("MAE = {}", metrics.meanAbsoluteError());
		// Explained variance
		log.info("Explained Variance = {}", metrics.explainedVariance());
	}

	public void outputPrepare(String outputPath) {
		try {
			File dir = new File(outputPath);
			if (dir.exists() == false)
				dir.mkdirs();
			FileUtils.cleanDirectory(new File(outputPath));
		} catch (IOException e) {
			log.error("Fail to delete clean dir：{}", e);
		}
	}

}
