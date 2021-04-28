package gof5.spark;

import java.util.List;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.LoggerFactory;

import gof5.spark.regression.strategy.DecisionTreeStrategy;
import gof5.spark.regression.strategy.GradientBoostedTreesStrategy;
import gof5.spark.regression.strategy.LinearRegressionStrategy;
import gof5.spark.regression.strategy.RandomForestsStrategy;

public class HousePricePredictML {

	private static final org.slf4j.Logger log = LoggerFactory.getLogger(HousePricePredictML.class);

	public static void main(String[] args) {

		// 1. Setting the Spark Context
		// 2. Loading the Data-set

		JavaRDD<String> data;
		String header;
		SparkConf conf = new SparkConf().setAppName("Main").setMaster("local[2]").set("spark.executor.memory", "3g")
				.set("spark.driver.memory", "3g");
		JavaSparkContext sc = new JavaSparkContext(conf);

		if("gcp".equals(args[0])){
			SparkSession spark = SparkSession.builder().appName("Main").getOrCreate();
			Dataset<Row> rows = spark.read().format("bigquery").option("table", "1111.X_train").load().cache();
			for(String c: rows.columns()){
				rows = rows.withColumn(c, rows.col(c).cast("String"));
			}

			data = rows.toJavaRDD().map(r -> {
				StringBuilder input = new StringBuilder();
				//take 1 column to last column
				for(int i=1; i<r.length()-1; i++){
					input.append((r.getString(i))).append(",");
				}
				input.append((r.getString(r.length()-1)));

				String rowString = input.toString();
				//log.info("rowstring"+rowString);
				return rowString;
			});

			List<Row> rowList = rows.collectAsList();
			StringBuilder builder = new StringBuilder();
			for(int i=0; i<rowList.size()-1; i++){
				//take only 1 column
				String label=rowList.get(i).getString(0);
				builder.append(label).append(",");

			}
			builder.append(rowList.get(rowList.size()-1).getString(0));
			header=builder.toString();
			//log.info("header"+header);
		}
		else {

			Logger.getLogger("org").setLevel(Level.OFF);
			Logger.getLogger("akka").setLevel(Level.OFF);

			String inputFile = "D:\\Code\\Personal\\Team1_Data_Analytics\\selected.csv";

			data = sc.textFile(inputFile);
			header = data.first();
		}



		// 3. Exploratory Data Analysis
		// 3.1. Creating Vector of Input Data
		String finalHeader1 = header;
		JavaRDD<Vector> inputData = data.filter(line -> !line.equals(finalHeader1)).map(line -> {
			String[] parts = line.split(",");
			double[] v = new double[parts.length - 1];
			for (int i = 0; i < parts.length - 1; i++) {
				v[i] = Double.parseDouble(parts[i]);
			}
			return Vectors.dense(v);
		});
		// 3.2. Performing Statistical Analysis
		MultivariateStatisticalSummary summary = Statistics.colStats(inputData.rdd());
		log.info("Summary Mean: " + summary.mean().toString());
		log.info("Summary Variance: " + summary.variance().toString());
		log.info("Summary Non-zero: " + summary.numNonzeros().toString());
		// 3.3. Performing Correlation Analysis
		Matrix correlMatrix = Statistics.corr(inputData.rdd(), "pearson");
		log.info("Correlation Matrix: " + correlMatrix.toString());

		// 4. Data Preparation
		// 4.1 Creating LabeledPoint of Input and Output Data
		String finalHeader = header;
		JavaRDD<LabeledPoint> parsedData = data.filter(line -> !line.equals(finalHeader)).map(line -> {
			String[] parts = line.split(",");
			double[] v = new double[parts.length - 1];
			for (int i = 0; i < parts.length - 1; i++) {
				v[i] = Double.parseDouble(parts[i]);
			}
			return new LabeledPoint(Double.parseDouble(parts[parts.length - 1]), Vectors.dense(v));
		});

		// 5. Data Splitting into 80% Training and 20% Test Sets
		JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[] { 0.8, 0.2 }, 11L);
		JavaRDD<LabeledPoint> trainingData = splits[0].cache();
		JavaRDD<LabeledPoint> testData = splits[1];

		// 6. Modeling
		String basePath = "D:\\Code\\Personal\\BigData-HousePricePrediction\\src\\main\\resources\\model\\";

		if("gcp".equals(args[0])){
			basePath = "gs://machine-learning-model-swe5003/";
		}

		// 6.1 - LinearRegressionWithSGD
		String output_lr = basePath + "LinearRegressionWithSGD";
		log.info("LRStrategy Model");
		LinearRegressionStrategy lrStrategy = new LinearRegressionStrategy();
		lrStrategy.execute(sc, trainingData, testData, output_lr);

		// 6.2 - DecisionTree
		String output_dt = basePath + "DecisionTree";
		log.info("DecisionTree Model");
		DecisionTreeStrategy dtStrategy = new DecisionTreeStrategy();
		dtStrategy.execute(sc, trainingData, testData, output_dt);

		// 6.3 - RandomForests
		String output_rf = basePath + "RandomForests";
		log.info("RandomForests Model");
		RandomForestsStrategy rfStrategy = new RandomForestsStrategy();
		rfStrategy.execute(sc, trainingData, testData, output_rf);

		// 6.4 - GradientBoostedTrees
		String output_gbt = basePath + "GradientBoostedTrees";
		log.info("GradientBoostedTrees Model");
		GradientBoostedTreesStrategy gbtStrategy = new GradientBoostedTreesStrategy();
		gbtStrategy.execute(sc, trainingData, testData, output_gbt);

		// 7. Clean-up
		sc.stop();
		sc.close();
	}

}