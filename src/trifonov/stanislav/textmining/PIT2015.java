package trifonov.stanislav.textmining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.xeiam.xchart.BitmapEncoder;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Histogram;
import com.xeiam.xchart.BitmapEncoder.BitmapFormat;
import com.xeiam.xchart.StyleManager.ChartType;
import com.xeiam.xchart.StyleManager.LegendPosition;

import opennlp.tools.stemmer.PorterStemmer;
import opennlp.tools.stemmer.Stemmer;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.util.HashList;

/**
 * A system that solves Semeval 2015 Task 1 - Paraphrase and Semantic Similarity in Twitter (PIT-2015)
 * 
 * @author stan0
 *
 */
public class PIT2015 {

	public static final String FILENAME_TRAIN = "train.data";
	public static final String FILENAME_DEV = "dev.data";
	public static final String FILENAME_TOKENIZER_MODEL = "en-token.bin";
	public static final String DIRNAME_DATA = "../SemEval-PIT2015-github/data";

	public static final int PARAPHRASE_NONE = -1;
	public static final int PARAPHRASE_DEBATABLE = 0;
	public static final int PARAPHRASE = 1;
	
	public static final Map<String, Integer> LABEL_TYPE = new HashMap<String, Integer>();
    
    
	private final File _fileTrain;
	
	private PIT2015() {
		_fileTrain = new File(DIRNAME_DATA, FILENAME_TRAIN);

		LABEL_TYPE.put("(3, 2)", PARAPHRASE);
		LABEL_TYPE.put("(4, 1)", PARAPHRASE);
		LABEL_TYPE.put("(5, 0)", PARAPHRASE);
		
		LABEL_TYPE.put("(1, 4)", PARAPHRASE_NONE);
		LABEL_TYPE.put("(0, 5)", PARAPHRASE_NONE);
		
		LABEL_TYPE.put("(2, 3)", PARAPHRASE_DEBATABLE);
	}
	
	public static void main(String[] args) {

		PIT2015 pit2015 = new PIT2015();
		try {
			pit2015.train();
		} catch (IOException e) {
			System.out.println("Training failed: " + e.getMessage());
			e.printStackTrace();
		}
	}
	
	private void train() throws IOException {
		BufferedReader reader = null;
		long start = System.currentTimeMillis();
		try {
			reader = new BufferedReader(new FileReader(_fileTrain));
			String line = null;
			String s1, s2, label, s1tag, s2tag;
			String columns[];
			List<Map<String, Double>> allItemsFeatures = new ArrayList<Map<String,Double>>();
			List<Integer> paraphraseLabels = new ArrayList<Integer>();
			String featureName = null;
			double featureValue = 0.0;
			Map<String, Double> maxFeatureValues = new HashMap<String, Double>();
			Map<String, Double> minFeatureValues = new HashMap<String, Double>();

			while( (line=reader.readLine()) != null ) {
				columns = line.split("\t");
				
				s1 = columns[2];
				s2 = columns[3];
				label = columns[4];
				s1tag = columns[5];
				s2tag = columns[6];
				
//				for(String c : columns)
//					System.out.println(c);
				
				Map<String, Double> features = nGramsOverlap(s1, s2);
				allItemsFeatures.add( features );
				paraphraseLabels.add( LABEL_TYPE.get(label) );
				
				for(Map.Entry<String, Double> entry : features.entrySet()) {
					featureName = entry.getKey();
					featureValue = entry.getValue().doubleValue();
//					System.out.println(featureName + ": " + featureValue);
					
					if(maxFeatureValues.get(featureName) == null ||
							maxFeatureValues.get(featureName) < featureValue)
						maxFeatureValues.put(featureName, Double.valueOf(featureValue));
					
					if(minFeatureValues.get(featureName) == null ||
							minFeatureValues.get(featureName) > featureValue)
						minFeatureValues.put(featureName, Double.valueOf(featureValue));
				}
				
//				break;
			}
			
			long end = System.currentTimeMillis();
			
			System.out.println("Items found: " + allItemsFeatures.size() + " in " + (end-start) + "ms.");
			
//			for(Map.Entry<String, Double> entry : maxFeatureValues.entrySet())
//				System.out.println("max " + entry.getKey() + " = " + entry.getValue());
//			for(Map.Entry<String, Double> entry : minFeatureValues.entrySet())
//				System.out.println("min " + entry.getKey() + " = " + entry.getValue());
			
			start = System.currentTimeMillis();
			
			File chartsDir = new File("featuresCharts");
			chartsDir.mkdirs();
			
			for(String key : maxFeatureValues.keySet()) {
				List<Double> featureValuesForParaphrases = new ArrayList<Double>();
				List<Double> featureValuesForNonparaphrases = new ArrayList<Double>();
				
				for(int i=0; i<allItemsFeatures.size(); ++i) {
					Map<String,Double> featureMap = allItemsFeatures.get(i);
					if(paraphraseLabels.get(i) == PARAPHRASE)
						featureValuesForParaphrases.add(featureMap.get(key));
					else if(paraphraseLabels.get(i) == PARAPHRASE_NONE)
						featureValuesForNonparaphrases.add(featureMap.get(key));
				}
				
				Histogram hParaphrases = new Histogram(
						featureValuesForParaphrases,
						100,
						minFeatureValues.get(key),
						maxFeatureValues.get(key));
				Histogram hNonparaphrases = new Histogram(
						featureValuesForNonparaphrases,
						100,
						minFeatureValues.get(key),
						maxFeatureValues.get(key));
				
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
			end = System.currentTimeMillis();
			System.out.println("Exporting features charts took " + (end-start) + "ms.");
		}
		finally {
			reader.close();
		}
	}
	
	private void tfidf() {
		
	}
	
	/**
	 * Computes features based on 1,2 and 3-grams overlapping.
	 * Implements the given baseline (linear regression of simple semantic features).
	 */
	private Map<String, Double> nGramsOverlap(String s1, String s2) {
		SimpleTokenizer tokenizer = SimpleTokenizer.INSTANCE;

		String s11grams[] = tokenizer.tokenize(s1);
		String s21grams[] = tokenizer.tokenize(s2);
		
		Stemmer stemmer = new PorterStemmer();
		List<String> s11stems = new ArrayList<String>();
		List<String> s21stems = new ArrayList<String>();

		for(String word : s11grams)
			s11stems.add( stemmer.stem(word).toString() );
		for(String word : s21grams)
			s21stems.add( stemmer.stem(word).toString() );
		
		Map<String, Double> features = new HashMap<String, Double>();
		
		double[] ngramFeatures = ngramFeatures(s11grams, s21grams);
		double[] ngramStemFeatures = ngramFeatures(
				s11stems.toArray(new String[s11stems.size()]),
				s21stems.toArray(new String[s21stems.size()]) );
		
		features.put("precision1gram", ngramFeatures[0]);
		features.put("recall1gram", ngramFeatures[1]);
		features.put("f1gram", ngramFeatures[2]);
		features.put("precision2gram", ngramFeatures[3]);
		features.put("recall2gram", ngramFeatures[4]);
		features.put("f2gram", ngramFeatures[5]);
		features.put("precision3gram", ngramFeatures[6]);
		features.put("recall3gram", ngramFeatures[7]);
		features.put("f3gram", ngramFeatures[8]);
		
		features.put("precision1stem", ngramStemFeatures[0]);
		features.put("recall1stem", ngramStemFeatures[1]);
		features.put("f1stem", ngramStemFeatures[2]);
		features.put("precision2stem", ngramStemFeatures[3]);
		features.put("recall2stem", ngramStemFeatures[4]);
		features.put("f2stem", ngramStemFeatures[5]);
		features.put("precision3stem", ngramStemFeatures[6]);
		features.put("recall3stem", ngramStemFeatures[7]);
		features.put("f3stem", ngramStemFeatures[8]);
		
		return features;
	}
	
	private double[] ngramFeatures(String s1Words[], String s2Words[]) {
		List<String> s12grams = new ArrayList<String>();
		List<String> s22grams = new ArrayList<String>();
		List<String> s13grams = new ArrayList<String>();
		List<String> s23grams = new ArrayList<String>();
		
		for (int i = 0; i < s1Words.length-1; i++)
			s12grams.add( s1Words[i] + " " + s1Words[i+1]);
		for (int i = 0; i < s1Words.length-2; i++)
			s13grams.add( s1Words[i] + " " + s1Words[i+1] + " " + s1Words[i+2]);
		
		for (int i = 0; i < s2Words.length-1; i++)
			s22grams.add( s2Words[i] + " " + s2Words[i+1]);
		for (int i = 0; i < s2Words.length-2; i++)
			s23grams.add( s2Words[i] + " " + s2Words[i+1] + " " + s2Words[i+2]);
		
		
		/*for(int i=0; i<s1Words.length; ++i)
			System.out.print(s1Words[i] + " ");
		System.out.println();
		for(int i=0; i<s12grams.size(); ++i)
			System.out.print(s12grams.get(i) + "; ");
		System.out.println();
		for(int i=0; i<s13grams.size(); ++i)
			System.out.print(s13grams.get(i) + "; ");
		System.out.println();*/
		
		int commonItemsCount = numberOfCommonItems(s1Words, s2Words);
		double precision1gram = commonItemsCount / (double)s1Words.length;
		double recall1gram = commonItemsCount / (double)s2Words.length;
		double f1gram = 0;
		if (precision1gram + recall1gram > 0)
			f1gram = 2 * precision1gram * recall1gram / (double)(precision1gram + recall1gram);
		
		commonItemsCount = numberOfCommonItems(s12grams, s22grams);
		double precision2gram = commonItemsCount / (double)s12grams.size();
		double recall2gram = commonItemsCount / (double)s22grams.size();
		double f2gram = 0;
		if(precision2gram + recall2gram > 0)
			f2gram = 2 * precision2gram * recall2gram / (double)(precision2gram + recall2gram);
		
		commonItemsCount = numberOfCommonItems(s13grams, s23grams);
		double precision3gram = commonItemsCount / (double)s13grams.size();
		double recall3gram = commonItemsCount / (double)s23grams.size();
		double f3gram = 0;
		if(precision3gram + recall3gram > 0)
			f3gram = 2 * precision3gram * recall3gram / (double)(precision3gram + recall3gram);
		
		
		return new double[] {
				precision1gram, recall1gram, f1gram,
				precision2gram, recall2gram, f2gram,
				precision3gram, recall3gram, f3gram };
	}
	
	private int numberOfCommonItems(List<String> l, List<String> r) {
		return numberOfCommonItems(
				l.toArray(new String[l.size()]),
				r.toArray(new String[r.size()]) );
	}
	private int numberOfCommonItems(String l[], String r[]) {
		int count = 0;
		for(int i=0; i<l.length; ++i) {
			for(int j=0; j<r.length; ++j)
				if(l[i].equalsIgnoreCase(r[j]))
					++count;
		}
		
		return count;
	}
	
	
	private void wordOrderSimilarity() {
		
	}
	
	/**
	 * 3.3.3 The Combined Semantic and Syntactic Measures 
	 */
	private void semanticSyntacticSimilarity() {
		
	}
}
