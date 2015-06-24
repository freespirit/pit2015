package trifonov.stanislav.textmining;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import com.xeiam.xchart.BitmapEncoder;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Histogram;
import com.xeiam.xchart.BitmapEncoder.BitmapFormat;
import com.xeiam.xchart.StyleManager.ChartType;
import com.xeiam.xchart.StyleManager.LegendPosition;

import trifonov.stanislav.ml.ClusteringKMeansModel;
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
	private static final float LABEL_PREDICTION_BORDER = 0.4f;
	private final static long ONE_GB = 1024 * 1024 * 1024;

	public static void main(String[] args) throws IOException, InterruptedException {		
			File fileTrain = new File(DIRNAME_DATA, FILENAME_TRAIN);
			File fileTest = new File(DIRNAME_DATA, FILENAME_TEST);
			File fileTestLabel = new File(DIRNAME_DATA, FILENAME_TEST_LABEL);
			File fileDev = new File(DIRNAME_DATA, FILENAME_DEV);
			String outputFileNameFormat = "PIT2015_STAN_01_%s.output";
			
			PIT2015 pit2015 = new PIT2015();
			pit2015.initW2VModel(fileTrain);

			Map<String, IMLModel> models = new HashMap<String, IMLModel>();
			models.put( "regrrun", new RegressionModel() );
			for(int k=4; k<=4; ++k)
				models.put( k+"means", new ClusteringKMeansModel(k, 1.1) );
			
			for(Entry<String, IMLModel> entry : models.entrySet()) {
				System.out.println();
				System.out.println(entry.getKey());
				pit2015.setModel( entry.getValue() );
				pit2015.trainWithDataFile(fileTrain);
				pit2015.evaluate(fileDev);
				File fileOutput = new File(DIRNAME_OUTPUT, String.format(outputFileNameFormat, entry.getKey()));
				pit2015.predictAndExport(fileTest, fileOutput);
				PIT2015.evalWithScripts(fileTestLabel, fileOutput);
				
//				System.out.println(String.format("entropy: %.3f \tpurity: %.3f", model.getEntropy(), model.getPurity()));
//				System.out.println( String.format("clustering score: %.3f", model.evaluate()) );
			}
			
			exportFeaturesCharts(pit2015._trainingPairData);
	}
	
	public static class FeatureMap extends HashMap<String, Double> {
		private static final long serialVersionUID = 1L;
	}
	
	public static final String FILENAME_TRAIN = "train.data";
	public static final String FILENAME_DEV = "dev.data";
	public static final String FILENAME_TEST = "test.data";
	public static final String FILENAME_TEST_LABEL = "test.label";
	public static final String FILENAME_TOKENIZER_MODEL = "en-token.bin";
	public static final String FILENAME_WORD2VEC_BIN = "GoogleNews-vectors-negative300.bin";
	
	public static final String DIRNAME_DATA = "../SemEval-PIT2015-github/data";
	public static final String DIRNAME_OUTPUT = "../output";
	public static final String DIRNAME_WORD2VEC_LOCATION = "/Volumes/storage/development/word2vec_stuff";

	public static final int COLUMN_INDEX_TOPICID = 0;
	public static final int COLUMN_INDEX_TOPIC = 1;
	public static final int COLUMN_INDEX_SENT1 = 2;
	public static final int COLUMN_INDEX_SENT2 = 3;
	public static final int COLUMN_INDEX_LABEL = 4;
	public static final int COLUMN_INDEX_SENT1TAG = 5;
	public static final int COLUMN_INDEX_SENT2TAG = 6;
	
	public static final Map<String, Float> LABEL_TYPE = new HashMap<String, Float>();
    
    private IMLModel _model;
    private FeaturesExtractor _featuresExtractor;
	private final List<PairData> _trainingPairData = new ArrayList<PairData>();
	private Map<String, float[]> _word2vecs;
	
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
	
	private PairData pairData(String s1Tags, String s2Tags, String label) throws IOException {
		if(_featuresExtractor == null)
			_featuresExtractor = new FeaturesExtractor(s1Tags, s2Tags, _word2vecs);
		else
			_featuresExtractor.init(s1Tags, s2Tags);
		
		List<Feature> features = new ArrayList<Feature>();
		features.add(_featuresExtractor.getWordOrderSimilarity());
		features.add(_featuresExtractor.getSemanticSimilarity());
		features.add(_featuresExtractor.getWord2VecFeature());
		features.add(_featuresExtractor.getW2VSSFeature());
		features.add(_featuresExtractor.getW2VCosSimFeature());
		
		features.add(_featuresExtractor.get1gramPrecision());
		features.add(_featuresExtractor.get1gramRecall());
		features.add(_featuresExtractor.get1gramF1());
		features.add(_featuresExtractor.get1gramStemPrecision());
		features.add(_featuresExtractor.get1gramStemRecall());
		features.add(_featuresExtractor.get1gramStemF1());
		
		features.add(_featuresExtractor.get2gramPrecision());
		features.add(_featuresExtractor.get2gramRecall());
		features.add(_featuresExtractor.get2gramF1());
		features.add(_featuresExtractor.get2gramStemPrecision());
		features.add(_featuresExtractor.get2gramStemRecall());
		features.add(_featuresExtractor.get2gramStemF1());
		
		features.add(_featuresExtractor.get3gramPrecision());
		features.add(_featuresExtractor.get3gramRecall());
		features.add(_featuresExtractor.get3gramF1());
		features.add(_featuresExtractor.get3gramStemPrecision());
		features.add(_featuresExtractor.get3gramStemRecall());
		features.add(_featuresExtractor.get3gramStemF1());
		
		return new PairData(LABEL_TYPE.get(label), features);
	}
	
	public Map<String, float[]> load_word2vec_fromFile(Collection<String> words) throws IOException {
		Map<String, float[]> word2vecs = new HashMap<String, float[]>();
		
		File inFile = new File(
				DIRNAME_WORD2VEC_LOCATION,
				FILENAME_WORD2VEC_BIN );
		
		FileInputStream is = null;
		try {
			is = new FileInputStream(inFile);
			final FileChannel channel = is.getChannel();
			MappedByteBuffer buffer = channel.map(MapMode.READ_ONLY, 0, Integer.MAX_VALUE);
			buffer.order(ByteOrder.LITTLE_ENDIAN);
			int bufferCount = 1;
			
			StringBuilder sb = new StringBuilder();
			char c = (char) buffer.get();
			while (c != '\n') {
				sb.append(c);
				c = (char) buffer.get();
			}
			
			String firstLine = sb.toString();
			int index = firstLine.indexOf(' ');
	
			final int vocabSize = Integer.parseInt(firstLine.substring(0, index));
			final int layerSize = Integer.parseInt(firstLine.substring(index + 1));
			
			System.out.println( vocabSize + " " + layerSize);

			final float[] floats = new float[layerSize];
			long start = System.currentTimeMillis();
			
			for(int lineNumber = 0; lineNumber < vocabSize; ++lineNumber) {
				sb.setLength(0);
				c = (char) buffer.get();
				while (c != ' ') {
					// ignore newlines in front of words (some binary files have newline,
					// some don't)
					if (c != '\n') {
						sb.append(c);
					}
					c = (char) buffer.get();
				}
				
				// read vector
				final FloatBuffer floatBuffer = buffer.asFloatBuffer();
				floatBuffer.get(floats);
				buffer.position(buffer.position() + 4 * layerSize);
				
				if( words.contains(sb.toString()) ) {
					String word = sb.toString();
					float[] word2vec = Arrays.copyOf(floats, floats.length);
					word2vecs.put(word, word2vec);
				}
				
				// remap file
				if (buffer.position() > ONE_GB) {
					final int newPosition = (int) (buffer.position() - ONE_GB);
					final long size = Math.min(channel.size() - ONE_GB * bufferCount, Integer.MAX_VALUE);
					System.out.println(
							String.format(
									"Reading gigabyte #%d. Start: %d, size: %d",
									bufferCount,
									ONE_GB * bufferCount,
									size));
					buffer = channel.map( FileChannel.MapMode.READ_ONLY, ONE_GB * bufferCount, size);
					buffer.order(ByteOrder.LITTLE_ENDIAN);
					buffer.position(newPosition);
					bufferCount += 1;
				}
			}
			
			System.out.println("Loading " + words.size() + " word2vecs took " + (System.currentTimeMillis()-start) + "ms.");
			System.out.println("" + word2vecs.size() + " word2vecs found");
		}
		finally {
			is.close();
		}
		
		return word2vecs;
	}
	
	
	/**
	 * Simply read the bin file and load the vectors for each word found in the corpus (all tweets)
	 * @param dataFile
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public void initW2VModel(File dataFile) throws IOException, InterruptedException {
		BufferedReader reader = null;
		try {
			reader = new BufferedReader( new FileReader(dataFile) );
			String lineInFile = null;
			Set<String> words = new HashSet<String>();
			
			while( (lineInFile=reader.readLine()) != null ) {
				String[] columns = lineInFile.split("\t");

				String tags[] = columns[COLUMN_INDEX_SENT1TAG].split(" ");
				for(int i=0; i<tags.length; ++i)
					words.add( tags[i].substring(0, tags[i].indexOf('/')) );
				
				tags = columns[COLUMN_INDEX_SENT2TAG].split(" ");
				for(int i=0; i<tags.length; ++i)
					words.add( tags[i].substring(0, tags[i].indexOf('/')) );
			}
			
			_word2vecs = load_word2vec_fromFile(words);
			
		}
		finally {
			reader.close();
		}
	}
	
	public void trainWithDataFile(File dataFile) throws IOException {
		long start = System.currentTimeMillis();
//		_trainingPairData.clear();
		if(_trainingPairData == null || _trainingPairData.isEmpty()) {
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
					_trainingPairData.add( pd );
				}
				
			} finally {
				if(reader != null)
					reader.close();
			}
		}
		
		for(PairData pd : _trainingPairData) {
			List<Feature> features = pd.getFeatures();
			double x[] = new double[features.size()];
			for (int i = 0; i < features.size(); i++)
				x[i] = features.get(i)._featureValue.doubleValue();
			feed(x, pd.getLabel());
		}
		
		_model.build();
		
		long end = System.currentTimeMillis();
		System.out.println("Trained in " + (end-start) + "ms." + "\tItems found: " + _trainingPairData.size());
	}
	
	public void evaluate(File testData) throws IOException {
		BufferedReader dataReader = null;
		List<Double> estimations = new ArrayList<Double>();
		List<Float> labels = new ArrayList<Float>();
		
		long start = System.currentTimeMillis();
		
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
					if(estimation >= LABEL_PREDICTION_BORDER)
						truePositives++;
					else if(estimation < PairData.LABEL_DEBATABLE)
						falseNegatives++;
				}
				else if( labelValue < PairData.LABEL_DEBATABLE ) {
					if(estimation >= LABEL_PREDICTION_BORDER)
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
			
			System.out.println(
					String.format(
							"%.3f\t%.3f\t%.3f\ttime:%.3f",
							f1,
							precision,
							recall,
							(System.currentTimeMillis()-start)/1000f ));
			
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
				
				//838	STAN	01_regrrun		0.612	0.625	0.600		0.525	0.627	0.573	0.691 with regression and (estimation > 0.4f ? true :false) on test.data
				double estimation = estimate(x);
				String resultLabel = (estimation >= LABEL_PREDICTION_BORDER ? "true" : "false");
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
	
	public static void evalWithScripts(File testFile, File outputFile) throws IOException {
		List<String> command = new ArrayList<String>();
		command.add("python");
		command.add(DIRNAME_OUTPUT + File.separator + "pit2015_eval_single.py");
		command.add(DIRNAME_OUTPUT + File.separator + testFile.getName());
		command.add(DIRNAME_OUTPUT + File.separator + outputFile.getName());
		
		long start = System.currentTimeMillis();

		Process p = new ProcessBuilder(command).start();
		BufferedReader reader = new BufferedReader( new InputStreamReader(p.getInputStream()) );
		String line;
		while( (line=reader.readLine()) != null )
			System.out.println(line);
		reader.close();
		
		reader = new BufferedReader( new InputStreamReader(p.getErrorStream()) );
		while( (line=reader.readLine()) != null )
			System.out.println(line);
		reader.close();
		
		System.out.println("evaluated in " + (System.currentTimeMillis() - start) + " ms.");
	}
	
/*	public static void exportDevLabels() throws IOException {
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
*/
}
