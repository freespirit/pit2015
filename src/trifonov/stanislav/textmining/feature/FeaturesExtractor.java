package trifonov.stanislav.textmining.feature;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import opennlp.tools.stemmer.PorterStemmer;
import opennlp.tools.stemmer.Stemmer;

public class FeaturesExtractor {

	public String _sentence1Tags;
	public String _sentence2Tags;
	
	private float[] _ngramFeatures = null;
	private float[] _ngramStemFeatures = null;
	
	private List<String> _s1Words = new ArrayList<String>();
	private List<String> _s2Words = new ArrayList<String>();
	
	private List<String> _s1POSTags = new ArrayList<String>();
	private List<String> _s2POSTags = new ArrayList<String>();
	private final Map<String, float[]> _word2vecs;
	
	public FeaturesExtractor(String tags1, String tags2, Map<String, float[]> word2vecs) {
		_word2vecs = word2vecs;
		
		init(tags1, tags2);
	}
	
	public void init(String tags1, String tags2) {
		_sentence1Tags = tags1;
		_sentence2Tags = tags2;
		_s1Words.clear();
		_s2Words.clear();
		_s1POSTags.clear();
		_s2POSTags.clear();
		
		String tags[] = _sentence1Tags.split(" ");
		for(int i=0; i<tags.length; ++i) {
			_s1Words.add( tags[i].substring(0, tags[i].indexOf('/')) );
			_s1POSTags.add( tags[i].split("/")[2] );
		}
		
		tags = _sentence2Tags.split(" ");
		for(int i=0; i<tags.length; ++i) {
			_s2Words.add( tags[i].substring(0, tags[i].indexOf('/')) );
			_s2POSTags.add( tags[i].split("/")[2] );
		}

		_ngramFeatures = null;
		_ngramStemFeatures = null;
	}
	
	public Feature getWordOrderSimilarity() {
		
		return new Feature( "wordOrder", new Float(wordOrderSimilarity()) );
	}
	
	private float wordOrderSimilarity() {
		List<String> allWords = new ArrayList<String>();
		
		for(String word : _s1Words)
			if( !allWords.contains(word) )
				allWords.add(word);
		for(String word : _s2Words)
			if( !allWords.contains(word) )
				allWords.add(word);
		
		int[] s1 = new int[allWords.size()];
		int[] s2 = new int[allWords.size()];
		
		for(int i=0; i<allWords.size(); ++i) {
			String word = allWords.get(i);
			s1[i] = _s1Words.contains(word) ? _s1Words.indexOf(word) : 0;
			s2[i] = _s2Words.contains(word) ? _s2Words.indexOf(word) : 0;
		}
		
		float sum = 0;
		float diff = 0;
		for(int i=0; i<allWords.size(); ++i) {
			sum += s1[i] + s2[i];
			diff += s1[i] - s2[i];
		}
		
		return 1 - (diff/sum);
	}

	// For WordOrder And Semantic similarity (vector):
	// http://ants.iis.sinica.edu.tw/3BkMJ9lTeWXTSrrvNoKNFDxRm3zFwRR/55/Sentence%20Similarity%20Based%20on%20Semantic%20Nets%20and%20corpus%20statistics.pdf
	//
	
	/**
	 * 3.3.3 The Combined Semantic and Syntactic Measures, the pdf 
	 */
	public Feature getSemanticSimilarity() {
		List<String> allWords = new ArrayList<String>();
		
		for(String word : _s1Words)
			if( !allWords.contains(word) )
				allWords.add(word);
		for(String word : _s2Words)
			if( !allWords.contains(word) )
				allWords.add(word);
		
		int[] s1 = new int[allWords.size()];
		int[] s2 = new int[allWords.size()];
		
		for(int i=0; i<allWords.size(); ++i) {
			s1[i] = _s1Words.contains(allWords.get(i)) ? 1 : 0;
			s2[i] = _s2Words.contains(allWords.get(i)) ? 1 : 0;
		}
		
		float lambda = 0.8f;
		
		float wo = wordOrderSimilarity();
		float cosSim = cosineSimilarity(s1, s2);
		float ssvwo = lambda * cosSim + (1-lambda) * wo;
		return new Feature( "ssv+wo", new Float(ssvwo) );
	}
	
	private float cosineSimilarity(int[] a, int[] b) {
		int dotProduct = 0;
		int magnitudeA = 0;
		int magnituteB = 0;
		
		for(int i=0; i<a.length; ++i) {
			dotProduct += a[i] * b[i];
			magnitudeA += a[i] * a[i];
			magnituteB += b[i] * b[i];
		}
		
		if(magnitudeA == 0 || magnituteB == 0)
			return 0;
		else
			return dotProduct / (float)(Math.sqrt(magnitudeA) * Math.sqrt(magnituteB));
	}
	
	private float cosineSimilarity(float[] a, float[] b) {
		int dotProduct = 0;
		int magnitudeA = 0;
		int magnituteB = 0;
		
		for(int i=0; i<a.length; ++i) {
			dotProduct += a[i] * b[i];
			magnitudeA += a[i] * a[i];
			magnituteB += b[i] * b[i];
		}
		
		if(magnitudeA == 0 || magnituteB == 0)
			return 0;
		else
			return dotProduct / (float)(Math.sqrt(magnitudeA) * Math.sqrt(magnituteB));
	}
	
	private double cosineSimilarity(double[] a, double[] b) {
		double dotProduct = 0;
		double magnitudeS1 = 0;
		double magnitudeS2 = 0;
		for(int i=0; i<a.length; ++i) {
			dotProduct += a[i] * b[i];
			magnitudeS1 += a[i] * a[i];
			magnitudeS2 += b[i] * b[i];
		}
		
		magnitudeS1 = Math.sqrt(magnitudeS1);
		magnitudeS2 = Math.sqrt(magnitudeS2);
		
		if(magnitudeS1 == 0 || magnitudeS2 ==0)
			return 0;
		else
			return dotProduct / (magnitudeS1 * magnitudeS2);
	}
	
	private double cosineSimilarity(List<RealVector> a, List<RealVector> b) {
		double dotProduct = 0;
		double magnitudeA = 0;
		double magnitudeB = 0;
		
		for(int i=0; i<a.size(); ++i) {
			dotProduct += a.get(i).dotProduct(b.get(i));
			magnitudeA += a.get(i).dotProduct(a.get(i));
			magnitudeB += b.get(i).dotProduct(b.get(i));
		}
		
		magnitudeA = Math.sqrt(magnitudeA);
		magnitudeB = Math.sqrt(magnitudeB);
		
		if(magnitudeA == 0 || magnitudeB == 0)
			return 0;
		else
			return dotProduct / (magnitudeA*magnitudeB);
	}
	
	private double maxW2VSimilarity(String word, String posTag, List<String> words, List<String> posTags) {
		double maxSimilarity = 0;
		
		for(int i=0; i<words.size(); ++i) {
			if( !posTags.get(i).equals(posTag) )
				continue;
			
			double similarity = 0;
			String candidateWord = words.get(i);
			if(_word2vecs.containsKey(words) && _word2vecs.containsKey(candidateWord)) {
				similarity = cosineSimilarity(_word2vecs.get(word), _word2vecs.get(candidateWord));
			}
			else
				similarity = Math.random();
			
			if(similarity > maxSimilarity)
				maxSimilarity = similarity;
		}
		
		return maxSimilarity;
	}
	
	/**
	 * Based on Malik et al. "Automatically Selecting Answer Templates to Respond to Customer Emails":
	 * Sum of max word similarities (word2vec cosine similarity) in the same POS class normalized by
	 * the sum of sentence lengths
	 * 
	 * @return
	 */
	public Feature getW2VSSFeature() {
		double s1SimSum = 0;
		double s2SimSum = 0;
		
		for(int i=0; i<_s1Words.size(); ++i) {
			String wordS1 = _s1Words.get(i);
			String posTag = _s1POSTags.get(i);
			s1SimSum += maxW2VSimilarity(wordS1, posTag, _s2Words, _s2POSTags);
		}
		
		for(int i=0; i<_s2Words.size(); ++i) {
			String wordS2 = _s2Words.get(i);
			String posTag = _s2POSTags.get(i);
			s2SimSum += maxW2VSimilarity(wordS2, posTag, _s1Words, _s1POSTags);
		}
		
		double lambda = 0.8;
		double score = (s1SimSum + s2SimSum) / (_s1Words.size() + _s2Words.size());
		double wo = wordOrderSimilarity();
		return new Feature( "semw2v", lambda*score + (1-lambda)*wo);
	}
	
	/**
	 * 
	 * Create a vector of all words (from the two sentences).
	 * For each sentence - the i-th element of its vector is the max word2vec similarity
	 * between that word and all the words in the sentence.
	 * @return 
	 */
	public Feature getW2VCosSimFeature() {
		List<String> allWords = new ArrayList<String>();
		List<String> allTags = new ArrayList<String>();
		
		for(int i=0; i< _s1Words.size(); ++i) {
			String word = _s1Words.get(i);
			if( !allWords.contains(word) ) {
				allWords.add(word);
				allTags.add(_s1POSTags.get(i));
			}
		}
		for(int i=0; i<_s2Words.size(); ++i) {
			String word = _s2Words.get(i);
			if( !allWords.contains(word) ) {
				allWords.add(word);
				allTags.add(_s2POSTags.get(i));
			}
		}
		
		double[] a = new double[allWords.size()];
		double[] b = new double[allWords.size()];
		
		for(int i=0; i<allWords.size(); ++i) {
			a[i] = maxW2VSimilarity(allWords.get(i), allTags.get(i), _s1Words, _s1POSTags);
			b[i] = maxW2VSimilarity(allWords.get(i), allTags.get(i), _s2Words, _s2POSTags);
		}
		
		double score = cosineSimilarity(a, b);
		return new Feature( "w2v_cos_sim", score );
	}
	
	public Feature getWord2VecFeature() throws IOException {
//		double[] s1Average = words2AverageVector(_s1Words, searcher);
//		double[] s2Average = words2AverageVector(_s2Words, searcher);
//
//		if(s1Average.length < s2Average.length)
//			s1Average = Arrays.copyOf(s1Average, s2Average.length);
//		else if(s2Average.length < s1Average.length)
//			s2Average = Arrays.copyOf(s2Average, s1Average.length);
//
//		
//		return new Feature( "word2vec", new Double(cosineSimilarity(s1Average, s2Average)) );
		
		List<String> allWords = new ArrayList<String>();
		
		for(String word : _s1Words)
			if( !allWords.contains(word) )
				allWords.add(word);
		for(String word : _s2Words)
			if( !allWords.contains(word) )
				allWords.add(word);
		
		List<RealVector> s1Vectors = new ArrayList<RealVector>(allWords.size());
		List<RealVector> s2Vectors = new ArrayList<RealVector>(allWords.size());
		double[] values = new double[300]; //ugly - that's the size of the word2vec
		double[] zeros = new double[300];
		
		for(int i=0; i<allWords.size(); ++i) {
			if(_word2vecs.containsKey(allWords.get(i))) {
				float[] w2v = _word2vecs.get(allWords.get(i));
				for(int j=0; j<values.length; ++j)
					values[j] = w2v[j];
			}
			else
				Arrays.fill(values, Math.random());
			
				
			
			if( _s1Words.contains(allWords.get(i)) )
				s1Vectors.add( new ArrayRealVector(values) );
			else
				s1Vectors.add( new ArrayRealVector(zeros) );
			
			if( _s2Words.contains(allWords.get(i)) )
				s2Vectors.add( new ArrayRealVector(values) );
			else
				s2Vectors.add( new ArrayRealVector(zeros) );
		}
		
		return new Feature( "word2vec_cossim", cosineSimilarity(s1Vectors, s2Vectors) );
	}
	
	@Deprecated
	private List<RealVector> getWV(List<String> words) {
		List<RealVector> wordVectors = new ArrayList<RealVector>();
		
		double[] values = new double[100];
		
		for(String word : words) {
			if(_word2vecs.containsKey(word)) {
				float[] w2v = _word2vecs.get(word);
				for(int i=0; i<100; ++i)
					values[i] = w2v[i];
			}
			else
				Arrays.fill(values, 0d);
			
			wordVectors.add( new ArrayRealVector(values) );
		}
		
		return wordVectors;
	}
	
	@Deprecated
	private double[] words2AverageVector(List<String> words) {
		List<float[]> wordVectors = new ArrayList<float[]>();
		int maxVectorSize = 0;
		
		for(String word : words)
			if(_word2vecs.containsKey(word)) {
				float[] vector = _word2vecs.get(word);
				if(vector.length > maxVectorSize)
					maxVectorSize = vector.length;
				wordVectors.add(vector);
//				System.out.println("\t" + word + " - " + searcher.getRawVector(word).toString());
			}
//			else
//				System.out.print(word + " ");//not found in word2vec model

//		System.out.println(maxVectorSize);
		double[] averageVector = new double[maxVectorSize];
		wordVectors.forEach(
					(float[] vector) ->
					{
						for(int i=0; i<vector.length; ++i)
							averageVector[i] = averageVector[i] + vector[i] /*/ (double)wordVectors.size()*/;
					}
				);
		
		return averageVector;
	}
	
	/**
	 * Computes features based on 1,2 and 3-grams overlapping.
	 * Implements the given baseline (linear regression of simple semantic features).
	 */
	private void prepareNGramOverlapFeatures() {
		if(_ngramFeatures == null || _ngramStemFeatures == null) {
			String tags[] = _sentence1Tags.split(" ");
			
			String s11grams[] = new String[tags.length];
			for(int i=0; i<tags.length; ++i)
				s11grams[i] = tags[i].substring(0, tags[i].indexOf('/'));
			
			tags = _sentence2Tags.split(" ");
			String s21grams[] = new String[tags.length];
			for(int i=0; i<tags.length; ++i)
				s21grams[i] = tags[i].substring(0, tags[i].indexOf('/'));
			
			
			Stemmer stemmer = new PorterStemmer();
			List<String> s11stems = new ArrayList<String>();
			List<String> s21stems = new ArrayList<String>();
	
			for(String word : s11grams)
				s11stems.add( stemmer.stem(word).toString() );
			for(String word : s21grams)
				s21stems.add( stemmer.stem(word).toString() );
			
			_ngramFeatures = nGramOverlaps(s11grams, s21grams);
			_ngramStemFeatures = nGramOverlaps(
					s11stems.toArray(new String[s11stems.size()]),
					s21stems.toArray(new String[s21stems.size()]) );
		}
		
//		"precision1gram", ngramFeatures[0]
//		"recall1gram", ngramFeatures[1]
//		"f1gram", ngramFeatures[2]
//		"precision2gram", ngramFeatures[3]
//		"recall2gram", ngramFeatures[4]
//		"f2gram", ngramFeatures[5]
//		"precision3gram", ngramFeatures[6]
//		"recall3gram", ngramFeatures[7]
//		"f3gram", ngramFeatures[8]
//		
//		"precision1stem", ngramStemFeatures[0]
//		"recall1stem", ngramStemFeatures[1]
//		"f1stem", ngramStemFeatures[2]
//		"precision2stem", ngramStemFeatures[3]
//		"recall2stem", ngramStemFeatures[4]
//		"f2stem", ngramStemFeatures[5]
//		"precision3stem", ngramStemFeatures[6]
//		"recall3stem", ngramStemFeatures[7]
//		"f3stem", ngramStemFeatures[8]
	}
	
	private float[] nGramOverlaps(String s1Words[], String s2Words[]) {
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
		System.out.print(s1Words[i] + "; ");
		System.out.println();
		for(int i=0; i<s12grams.size(); ++i)
			System.out.print(s12grams.get(i) + "; ");
		System.out.println();
		for(int i=0; i<s13grams.size(); ++i)
			System.out.print(s13grams.get(i) + "; ");
		System.out.println();*/
		
		int commonItemsCount = numberOfCommonItems(s1Words, s2Words);
		float precision1gram = commonItemsCount / (float)s1Words.length;
		float recall1gram = commonItemsCount / (float)s2Words.length;
		float f1gram = 0;
		if (precision1gram + recall1gram > 0)
			f1gram = 2 * precision1gram * recall1gram / (float)(precision1gram + recall1gram);
		
		commonItemsCount = numberOfCommonItems(s12grams, s22grams);
		float precision2gram = commonItemsCount / (float)s12grams.size();
		float recall2gram = commonItemsCount / (float)s22grams.size();
		float f2gram = 0;
		if(precision2gram + recall2gram > 0)
			f2gram = 2 * precision2gram * recall2gram / (float)(precision2gram + recall2gram);
		
		commonItemsCount = numberOfCommonItems(s13grams, s23grams);
		float precision3gram = commonItemsCount / (float)s13grams.size();
		float recall3gram = commonItemsCount / (float)s23grams.size();
		float f3gram = 0;
		if(precision3gram + recall3gram > 0)
			f3gram = 2 * precision3gram * recall3gram / (float)(precision3gram + recall3gram);
		
		return new float[] {
				precision1gram, recall1gram, f1gram,
				precision2gram, recall2gram, f2gram,
				precision3gram, recall3gram, f3gram };
	}
	
	private static int numberOfCommonItems(List<String> l, List<String> r) {
		return numberOfCommonItems(
				l.toArray(new String[l.size()]),
				r.toArray(new String[r.size()]) );
	}
	private static int numberOfCommonItems(String l[], String r[]) {
		int count = 0;
		for(int i=0; i<l.length; ++i) {
			for(int j=0; j<r.length; ++j)
				if(l[i].equalsIgnoreCase(r[j]))
					++count;
		}
		
		return count;
	}
	
	//1grams
	
	public Feature get1gramPrecision() {
		prepareNGramOverlapFeatures();
		return new Feature("1gramPrecision", new Float(_ngramFeatures[0]));
	}
	
	public Feature get1gramRecall() {
		prepareNGramOverlapFeatures();
		return new Feature("1gramRecall", new Float(_ngramFeatures[1]));
	}
	
	public Feature get1gramF1() {
		prepareNGramOverlapFeatures();
		return new Feature("1gramF1", new Float(_ngramFeatures[2]));
	}
	
	public Feature get1gramStemPrecision() {
		prepareNGramOverlapFeatures();
		return new Feature("1gramStemPrecision", new Float(_ngramStemFeatures[0]));
	}
	
	public Feature get1gramStemRecall() {
		prepareNGramOverlapFeatures();
		return new Feature("1gramStemRecall", new Float(_ngramStemFeatures[1]));
	}
	
	public Feature get1gramStemF1() {
		prepareNGramOverlapFeatures();
		return new Feature("1gramStemF1", new Float(_ngramStemFeatures[2]));
	}
	
	//2grams
	
	public Feature get2gramPrecision() {
		prepareNGramOverlapFeatures();
		return new Feature("2gramPrecision", new Float(_ngramFeatures[3]));
	}
	
	public Feature get2gramRecall() {
		prepareNGramOverlapFeatures();
		return new Feature("2gramRecall", new Float(_ngramFeatures[4]));
	}
	
	public Feature get2gramF1() {
		prepareNGramOverlapFeatures();
		return new Feature("2gramF1", new Float(_ngramFeatures[5]));
	}
	
	public Feature get2gramStemPrecision() {
		prepareNGramOverlapFeatures();
		return new Feature("2gramStemPrecision", new Float(_ngramStemFeatures[3]));
	}
	
	public Feature get2gramStemRecall() {
		prepareNGramOverlapFeatures();
		return new Feature("2gramStemRecall", new Float(_ngramStemFeatures[4]));
	}
	
	public Feature get2gramStemF1() {
		prepareNGramOverlapFeatures();
		return new Feature("2gramStemF1", new Float(_ngramStemFeatures[5]));
	}
	
	//3grams
	public Feature get3gramPrecision() {
		prepareNGramOverlapFeatures();
		return new Feature("3gramPrecision", new Float(_ngramFeatures[6]));
	}
	
	public Feature get3gramRecall() {
		prepareNGramOverlapFeatures();
		return new Feature("3gramRecall", new Float(_ngramFeatures[7]));
	}
	
	public Feature get3gramF1() {
		prepareNGramOverlapFeatures();
		return new Feature("3gramF1", new Float(_ngramFeatures[8]));
	}
	
	public Feature get3gramStemPrecision() {
		prepareNGramOverlapFeatures();
		return new Feature("3gramStemPrecision", new Float(_ngramStemFeatures[6]));
	}
	
	public Feature get3gramStemRecall() {
		prepareNGramOverlapFeatures();
		return new Feature("3gramStemRecall", new Float(_ngramStemFeatures[7]));
	}
	
	public Feature get3gramStemF1() {
		prepareNGramOverlapFeatures();
		return new Feature("3gramStemF1", new Float(_ngramStemFeatures[8]));
	}
}
