package trifonov.stanislav.textmining;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.commons.math3.stat.regression.AbstractMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import com.xeiam.xchart.BitmapEncoder;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Histogram;
import com.xeiam.xchart.BitmapEncoder.BitmapFormat;
import com.xeiam.xchart.StyleManager.ChartType;
import com.xeiam.xchart.StyleManager.LegendPosition;

import trifonov.stanislav.ml.IMLModel;
import trifonov.stanislav.ml.RegressionModel;
import trifonov.stanislav.textmining.feature.Feature;
import trifonov.stanislav.textmining.feature.FeaturesExtractor;

/**
 * A system that solves Semeval 2015 Task 1 - Paraphrase and Semantic Similarity in Twitter (PIT-2015)
 * 
 * @author stan0
 *
 */
public class PIT2015 {

	public static void main(String[] args) throws IOException {
/*//			exportDevLabels();
			pit2015.train(false);
//			pit2015.test();
			pit2015.evaluate();*/
		
			File fileTrain = new File(DIRNAME_DATA, FILENAME_TRAIN);
			File fileTest = new File(DIRNAME_DATA, FILENAME_TEST);
			File fileTestLabel = new File(DIRNAME_DATA, FILENAME_TEST_LABEL);
			File fileDev = new File(DIRNAME_DATA, FILENAME_DEV);
			File fileOutput = new File(DIRNAME_OUTPUT, "PIT2015_STAN_01_regrrun.output");
			
			PIT2015 pit2015 = new PIT2015();
			pit2015.setModel( new RegressionModel() );
			
			pit2015.trainWithDataFile(fileTrain);
			pit2015.evaluate(fileDev);
			
			pit2015.predictAndExport(fileTest, fileOutput);
			
//			exportFeaturesCharts(trainingPairsData);
	}
	
	public static class FeatureMap extends HashMap<String, Double> {
		private static final long serialVersionUID = 1L;
	}
	
	public static final String FILENAME_TRAIN = "train.data";
	public static final String FILENAME_DEV = "dev.data";
	public static final String FILENAME_TEST = "test.data";
	public static final String FILENAME_TEST_LABEL = "test.label";
	public static final String FILENAME_TOKENIZER_MODEL = "en-token.bin";
	public static final String DIRNAME_DATA = "../SemEval-PIT2015-github/data";
	public static final String DIRNAME_OUTPUT = "../output";

	public static final int COLUMN_INDEX_TOPICID = 0;
	public static final int COLUMN_INDEX_TOPIC = 1;
	public static final int COLUMN_INDEX_SENT1 = 2;
	public static final int COLUMN_INDEX_SENT2 = 3;
	public static final int COLUMN_INDEX_LABEL = 4;
	public static final int COLUMN_INDEX_SENT1TAG = 5;
	public static final int COLUMN_INDEX_SENT2TAG = 6;
	
	public static final Map<String, Float> LABEL_TYPE = new HashMap<String, Float>();
    
    private IMLModel _model;
	private final List<PairData> _trainingPairData = new ArrayList<PairData>();
	
	public PIT2015() {
		LABEL_TYPE.put("(5, 0)", PairData.LABEL_PARAPHRASE10);
		LABEL_TYPE.put("(4, 1)", PairData.LABEL_PARAPHRASE08);
		LABEL_TYPE.put("(3, 2)", PairData.LABEL_PARAPHRASE06);
		LABEL_TYPE.put("5", PairData.LABEL_PARAPHRASE10);
		LABEL_TYPE.put("4", PairData.LABEL_PARAPHRASE08);
		LABEL_TYPE.put("3", PairData.LABEL_PARAPHRASE06);
		
		LABEL_TYPE.put("(0, 5)", PairData.LABEL_NONPARAPHRASE00);
		LABEL_TYPE.put("(1, 4)", PairData.LABEL_NONPARAPHRASE02);
		LABEL_TYPE.put("0", PairData.LABEL_NONPARAPHRASE00);
		LABEL_TYPE.put("1", PairData.LABEL_NONPARAPHRASE02);
		
		LABEL_TYPE.put("(2, 3)", PairData.LABEL_DEBATABLE);
		LABEL_TYPE.put("2", PairData.LABEL_DEBATABLE);
		
	}
	
	private void setModel(IMLModel model) {
		_model = model;
	}
	
	private void feed(double data[], float label) {
		_model.feedData(data, label);
	}
	
	private double estimate(double data[]) {
		return _model.estimate(data);
	}
	
	private PairData pairData(String s1Tags, String s2Tags, String label) {
		FeaturesExtractor fe = new FeaturesExtractor(s1Tags, s2Tags);
		List<Feature> features = new ArrayList<Feature>();
//		features.add(fe.getWordOrderSimilarity());
//		features.add(fe.getSemanticSimilarity());
		
		features.add(fe.get1gramPrecision());
		features.add(fe.get1gramRecall());
		features.add(fe.get1gramF1());
		features.add(fe.get1gramStemPrecision());
		features.add(fe.get1gramStemRecall());
		features.add(fe.get1gramStemF1());
		
		features.add(fe.get2gramPrecision());
		features.add(fe.get2gramRecall());
		features.add(fe.get2gramF1());
		features.add(fe.get2gramStemPrecision());
		features.add(fe.get2gramStemRecall());
		features.add(fe.get2gramStemF1());
		
		features.add(fe.get3gramPrecision());
		features.add(fe.get3gramRecall());
		features.add(fe.get3gramF1());
		features.add(fe.get3gramStemPrecision());
		features.add(fe.get3gramStemRecall());
		features.add(fe.get3gramStemF1());
		
		return new PairData(LABEL_TYPE.get(label), features);
	}
	
	public void trainWithDataFile(File dataFile) throws IOException {
		long start = System.currentTimeMillis();
		_trainingPairData.clear();
		
		BufferedReader reader = null;
		try {
			reader = new BufferedReader( new FileReader(dataFile) );
			String lineInFile = null;
			
			while( (lineInFile=reader.readLine()) != null ) {
				String[] columns = lineInFile.split("\t");
				String label = columns[COLUMN_INDEX_LABEL];
				String tags1 = columns[COLUMN_INDEX_SENT1TAG];
				String tags2 = columns[COLUMN_INDEX_SENT2TAG];
				
				if(LABEL_TYPE.get(label) == PairData.LABEL_DEBATABLE)
					continue;
				
				PairData pd = pairData(tags1, tags2, label);
				
				List<Feature> features = pd.getFeatures();
				double x[] = new double[features.size()];
				for (int i = 0; i < features.size(); i++)
					x[i] = features.get(i)._featureValue.doubleValue();
				feed(x, pd.getLabel());
				
				_trainingPairData.add( pd );
			}
			
		} finally {
			if(reader != null)
				reader.close();
		}
		
		long end = System.currentTimeMillis();
		System.out.println("Trained in " + (end-start) + "ms." + "\tItems found: " + _trainingPairData.size());
	}
	
	public void evaluate(File testData) throws IOException {
		BufferedReader dataReader = null;
		List<Double> estimations = new ArrayList<Double>();
		List<Float> labels = new ArrayList<Float>();
		
		try {
			dataReader = new BufferedReader(new FileReader(testData));
			
			String dataLine = null, columns[], label, s1tags, s2tags;
			int truePositives = 0;
			int falsePositives = 0;
			int falseNegatives = 0;
			
			while( (dataLine=dataReader.readLine()) != null ) {
				columns = dataLine.split("\t");
				
				label = columns[COLUMN_INDEX_LABEL];
				s1tags = columns[COLUMN_INDEX_SENT1TAG];
				s2tags = columns[COLUMN_INDEX_SENT2TAG];
				
				PairData pairData = pairData(s1tags, s2tags, label);
				List<Feature> features = pairData.getFeatures();
				double x[] = new double[features.size()];
				for (int i = 0; i < x.length; i++)
					x[i] = features.get(i)._featureValue.doubleValue();
				
				double estimation = estimate(x);
				
				float labelValue = LABEL_TYPE.get(label);
				if(labelValue >= PairData.LABEL_PARAPHRASE06) {
					if(estimation >= PairData.LABEL_PARAPHRASE06)
						truePositives++;
					else if(estimation < PairData.LABEL_DEBATABLE)
						falseNegatives++;
				}
				else if( labelValue < PairData.LABEL_DEBATABLE ) {
					if(estimation >= PairData.LABEL_PARAPHRASE06)
						falsePositives++;
				}
				
				estimations.add(estimation);
//				if(estimation < PairData.LABEL_NONPARAPHRASE02)
//					estimations.add((double)PairData.LABEL_NONPARAPHRASE00);
//				else if(estimation < PairData.LABEL_DEBATABLE)
//					estimations.add((double)PairData.LABEL_NONPARAPHRASE02);
//				else if(estimation < PairData.LABEL_PARAPHRASE06)
//					estimations.add((double)PairData.LABEL_DEBATABLE);
//				else if(estimation < PairData.LABEL_PARAPHRASE08)
//					estimations.add((double)PairData.LABEL_PARAPHRASE06);
//				else if(estimation < PairData.LABEL_PARAPHRASE10)
//					estimations.add((double)PairData.LABEL_PARAPHRASE08);
//				else
//					estimations.add((double)PairData.LABEL_PARAPHRASE10);
				
				labels.add(labelValue);
//				System.out.println(labelValue +": " + estimation);
			}
			
			float precision = truePositives / (float)(truePositives+falsePositives);
			float recall = truePositives / (float)(truePositives+falseNegatives);
			float f1 = 2 * precision * recall / (precision + recall);
			
			System.out.println("F1: " + f1 + "\tprecision: " + precision + "\trecall: " + recall);
			
			Histogram hEstimations = new Histogram(estimations, 200);
//			Histogram hLabels = new Histogram(labels, 200);
			
			Chart chart = new ChartBuilder().chartType(ChartType.Bar)
					.width(800)
					.height(600)
					.title("Estimation")
					.xAxisTitle("Label")
					.yAxisTitle("Count")
					.build();
			chart.addSeries("Estimation", hEstimations.getxAxisData(), hEstimations.getyAxisData());
//			chart.addSeries("Original", hLabels.getxAxisData(), hLabels.getyAxisData());
			chart.getStyleManager().setLegendPosition(LegendPosition.InsideNE);
			chart.getStyleManager().setBarsOverlapped(true);
	
			BitmapEncoder.saveBitmap(chart,
					new File("0-Estimations").getAbsolutePath(),
					BitmapFormat.PNG);
		}
		finally {
			if (dataReader!=null)
				dataReader.close();
		}
	}
	
	public void predictAndExport(File dataFile, File outputFile) throws IOException {
		BufferedReader reader = null;
		BufferedWriter writer = null;
		
		try {
			reader = new BufferedReader( new FileReader(dataFile) );
			writer = new BufferedWriter( new FileWriter(outputFile) );
			String line, columns[], label, s1tags, s2tags;
			
			while( (line=reader.readLine()) != null ) {
				columns = line.split("\t");
				
				label = columns[COLUMN_INDEX_LABEL];
				s1tags = columns[COLUMN_INDEX_SENT1TAG];
				s2tags = columns[COLUMN_INDEX_SENT2TAG];
				
				PairData pd = pairData(s1tags, s2tags, label);
				List<Feature> features = pd.getFeatures();
				double x[] = new double[features.size()];
				for (int i = 0; i < x.length; i++)
					x[i] = features.get(i)._featureValue.doubleValue();
				
				double estimation = estimate(x);
				String resultLabel = (estimation > 0.4 ? "true" : "false");
				String resultScore = 
						String.format(
								Locale.US, "%.4f",
								Math.max( Math.min(estimation, 1.0), 0.0));
				
				writer.write(resultLabel + "\t" + resultScore);
				writer.newLine();
			}
		}finally {
			if(reader != null)
				reader.close();
			if(writer != null)
				writer.close();
		}
		
	}
	
	public static void exportFeaturesCharts(List<PairData> pairsData) throws IOException {
		Map<String, Number> maxFeaturesValues = new HashMap<String, Number>();
		Map<String, Number> minFeaturesValues = new HashMap<String, Number>();
		
		for(PairData pd : pairsData) {
			for(Feature f : pd.getFeatures()) {
				String featureName = f._featureName;
				float featureValue = f._featureValue.floatValue();
				
				if(maxFeaturesValues.get(featureName) == null || maxFeaturesValues.get(featureName).floatValue() < featureValue)
					maxFeaturesValues.put(featureName, Float.valueOf(featureValue));
				
				if(minFeaturesValues.get(featureName) == null || minFeaturesValues.get(featureName).floatValue() > featureValue)
					minFeaturesValues.put(featureName, Float.valueOf(featureValue));
			}
		}
		
		long start = System.currentTimeMillis();
		File chartsDir = new File("featuresCharts");
		chartsDir.mkdirs();
		for(String key : maxFeaturesValues.keySet()) {
			List<Number> featureValuesForParaphrases = new ArrayList<Number>();
			List<Number> featureValuesForNonparaphrases = new ArrayList<Number>();
			
			for(PairData pd : pairsData) {
				Feature f = pd.getFeature(key);
				float value = f._featureValue.floatValue();
				if(pd.getLabel() > PairData.LABEL_PARAPHRASE06)
					featureValuesForParaphrases.add(f._featureValue);
				else if(pd.getLabel() < PairData.LABEL_DEBATABLE)
					featureValuesForNonparaphrases.add(f._featureValue);
			}
			
			Histogram hParaphrases = new Histogram( featureValuesForParaphrases,
													100,
													minFeaturesValues.get(key).doubleValue(),
													maxFeaturesValues.get(key).doubleValue() );
			Histogram hNonparaphrases = new Histogram( featureValuesForNonparaphrases,
													   100,
													   minFeaturesValues.get(key).doubleValue(),
													   maxFeaturesValues.get(key).doubleValue() );
			
			Chart chart = new ChartBuilder()
						.chartType(ChartType.Bar)
						.width(800)
						.height(600)
						.title("Feature: " + key)
						.xAxisTitle(key)
						.yAxisTitle("Sentence pairs")
						.build();
			chart.addSeries("Paraphrases", hParaphrases.getxAxisData(), hParaphrases.getyAxisData());
			chart.addSeries("Non paraphrases", hNonparaphrases.getxAxisData(), hNonparaphrases.getyAxisData());
			chart.getStyleManager().setLegendPosition(LegendPosition.InsideNE);
			chart.getStyleManager().setBarsOverlapped(true);

			BitmapEncoder.saveBitmap(chart,
					new File(chartsDir, key).getAbsolutePath(),
					BitmapFormat.PNG);
		}
		long end = System.currentTimeMillis();
		System.out.println("Exporting features charts took " + (end-start) + "ms.");
	}
	
	public static void exportDevLabels() throws IOException {
		BufferedReader reader = null;
		BufferedWriter writer = null;
		
		try {
			reader = new BufferedReader( new FileReader( new File(DIRNAME_DATA, FILENAME_DEV) ) );
			writer = new BufferedWriter( new FileWriter(new File(DIRNAME_DATA, "dev.label")) );
			String line, columns[], label;
			
			while( (line=reader.readLine()) != null ) {
				columns = line.split("\t");
				
				label = columns[COLUMN_INDEX_LABEL];
				int labelValue = label.charAt(1) - '0';
				
				String resultLabel = (labelValue > 2 ? "true" :
					(labelValue == 2 ? "----" : "false"));
				String resultScore = 
						String.format( Locale.US, "%.4f", labelValue * 0.2 );
				
				writer.write(resultLabel + "\t" + resultScore);
				writer.newLine();
			}
		}finally {
			if(reader != null)
				reader.close();
			if(writer != null)
				writer.close();
		}
		
	}
}
