package trifonov.stanislav.textmining;

import java.util.ArrayList;
import java.util.List;

import trifonov.stanislav.textmining.feature.Feature;

public class PairData {

	public static final int LABEL_PARAPHRASE = 1;
	public static final int LABEL_DEBATABLE = 0;
	public static final int LABEL_NONPARAPHRASE = -1;
	
	private final List<Feature> _features = new ArrayList<Feature>();
	private final int _label;
	
	public PairData(int label, List<Feature> features) {
		_label = label;
		for(Feature f : features)
			_features.add(f);
	}
	
	public int getLabel() {
		return _label;
	}
	
	public Feature getFeature(String name) {
		for(Feature f : _features)
			if(f._featureName.equals(name))
				return f;
		
		return null;
	}
	
	public List<Feature> getFeatures() {
		return _features;
	}
}
