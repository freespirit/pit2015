package trifonov.stanislav.textmining;

import java.util.ArrayList;
import java.util.List;

import trifonov.stanislav.textmining.feature.Feature;

public class PairData {

	public static final float LABEL_PARAPHRASE10 = 1f;
	public static final float LABEL_PARAPHRASE08 = 0.8f;
	public static final float LABEL_PARAPHRASE06 = 0.6f;
	public static final float LABEL_DEBATABLE = 0.4f;
	public static final float LABEL_NONPARAPHRASE02 = 0.2f;
	public static final float LABEL_NONPARAPHRASE00 = 0f;
	
	private final List<Feature> _features = new ArrayList<Feature>();
	private final float _label;
	
	public PairData(float label, List<Feature> features) {
		_label = label;
		for(Feature f : features)
			_features.add(f);
	}
	
	public float getLabel() {
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
