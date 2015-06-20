package trifonov.stanislav.textmining.feature;

import java.util.ArrayList;
import java.util.List;

import opennlp.tools.stemmer.PorterStemmer;
import opennlp.tools.stemmer.Stemmer;

public class FeaturesExtractor {

	public final String _sentence1Tags;
	public final String _sentence2Tags;
	
	private float[] _ngramFeatures = null;
	private float[] _ngramStemFeatures = null;
	
	private List<String> _s1Words = new ArrayList<String>();
	private List<String> _s2Words = new ArrayList<String>();
	
	
	public FeaturesExtractor(String tags1, String tags2) {
		_sentence1Tags = tags1;
		_sentence2Tags = tags2;
		
		String tags[] = _sentence1Tags.split(" ");
		for(int i=0; i<tags.length; ++i)
			_s1Words.add( tags[i].substring(0, tags[i].indexOf('/')) );
		
		tags = _sentence2Tags.split(" ");
		for(int i=0; i<tags.length; ++i)
			_s2Words.add( tags[i].substring(0, tags[i].indexOf('/')) );
	}
	
	
	public Feature getWordOrderSimilarity() {
		Stemmer stemmer = new PorterStemmer();
		List<String> s1stems = new ArrayList<String>(_s1Words.size());
		List<String> s2stems = new ArrayList<String>(_s2Words.size());

		for(String word : _s1Words)
			s1stems.add( stemmer.stem(word).toString() );
		for(String word : _s2Words)
			s2stems.add( stemmer.stem(word).toString() );
		
		List<String> commonStems = new ArrayList<String>();
		for(String stem : s1stems)
			if(s2stems.contains(stem))
				commonStems.add(stem);
		
		int[] s1Order = new int[commonStems.size()];
		int[] s2Order = new int[commonStems.size()];
		
		for(int i=0; i<commonStems.size(); ++i) {
			String stem = commonStems.get(i);
			
			for(int j=0; j<s1stems.size(); ++j)
				if(s1stems.get(j).equals(stem)) {
					s1Order[i] = j;
					break;
				}
			
			for(int j=0; j<s2stems.size(); ++j)
				if(s2stems.get(j).equals(stem)) {
					s2Order[i] = j;
					break;
				}
		}
		
		double sum = 0;
		double diff = 0;
		for(int i=0; i<commonStems.size(); ++i) {
			sum += s1Order[i] + s2Order[i];
			diff += s1Order[i] - s2Order[i];
		}
		double wordOrderSimilarity = 1 - (diff/sum);
		return new Feature( "wordOrder", new Double(wordOrderSimilarity) );
	}
	
	/**
	 * 3.3.3 The Combined Semantic and Syntactic Measures, the pdf 
	 */
	public Feature getSemanticSimilarity() {
		int n = _s1Words.size() + _s2Words.size();
		
		return null;
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
