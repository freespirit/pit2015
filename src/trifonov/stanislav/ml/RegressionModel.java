package trifonov.stanislav.ml;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.regression.AbstractMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

public class RegressionModel implements IMLModel {

	private final List<double[]> _multipleRegressionData = new ArrayList<double[]>();
	AbstractMultipleLinearRegression _multipleRegression = new OLSMultipleLinearRegression();
	double[] _regressionParameters = null;
	
	@Override
	public void feedData(double[] data, float label) {
		double observation[] = new double[data.length + 1];
		observation[0] = label;
		for(int i=0; i<data.length; ++i)
			observation[i+1] = data[i];
		
		_multipleRegressionData.add(observation);
	}

	@Override
	public void build() {
		if(_multipleRegressionData.size() > 0) {
			int featuresCount = _multipleRegressionData.get(0).length - 1;
			double observations[] = new double[_multipleRegressionData.size() * (featuresCount+1)];
			for(int i=0; i<_multipleRegressionData.size(); ++i) {
				for(int j=0; j<_multipleRegressionData.get(i).length; ++j)
					observations[i*_multipleRegressionData.get(i).length + j] = _multipleRegressionData.get(i)[j];
			}
			_multipleRegression.setNoIntercept(true);
			_multipleRegression.newSampleData(observations, _multipleRegressionData.size(), featuresCount);
			_multipleRegressionData.clear();
		}
		
 		_regressionParameters = _multipleRegression.estimateRegressionParameters();
	}

	@Override
	public double estimate(double features[]) {
		
		
		double estimation = 0;
		for (int i = 0; i < features.length; i++)
			estimation += _regressionParameters[i] * features[i];

		return estimation;
	}

}
