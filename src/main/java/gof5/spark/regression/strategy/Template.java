package gof5.spark.regression.strategy;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.slf4j.LoggerFactory;

public abstract class Template {
	private static final org.slf4j.Logger log = LoggerFactory.getLogger(Template.class);

	public void ModelEvaluate(JavaPairRDD<Object, Object> rdd) {
		// 1 Accuracy
		MulticlassMetrics metrics = new MulticlassMetrics(rdd.rdd());
		double accuracy = metrics.accuracy();
		log.info("Model Accuracy on Test Data: " + accuracy);
		// 2 Mean Squared Error
		double MSE = rdd.mapToDouble(pair -> {
			double diff = (Double) pair._1() - (Double) pair._2();
			return diff * diff;
		}).mean();
		log.info("Mean Squared Error = " + MSE);
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
