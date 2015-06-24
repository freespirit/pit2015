package trifonov.stanislav.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.FuzzyKMeansClusterer;
import org.apache.commons.math3.ml.clustering.evaluation.ClusterEvaluator;
import org.apache.commons.math3.ml.clustering.evaluation.SumOfClusterVariances;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

import trifonov.stanislav.textmining.PairData;

public class ClusteringKMeansModel implements IMLModel {
	
	private static class Observation implements Clusterable {
		
		double[] _data;
		float _label;
		
		public Observation(double[] data, float label) {
			_data = data;
			_label = label;
		}

		@Override
		public double[] getPoint() {
			return _data;
		}
		
		public float getLabel() {
			return _label;
		}
	}

	private static class ClusterInfo {
		private final float _label;
		private final double _purity;
		private final double _entropy;
		
		public ClusterInfo(float label, double purity, double entropy) {
			_label = label;
			_purity = purity;
			_entropy = entropy;
		}
	}
	
	
	private final int _k;
	private final List<Observation> _points = new ArrayList<Observation>();
	private final FuzzyKMeansClusterer<Observation> _clusterer;
	private List<CentroidCluster<Observation>> _clusters;
	private List<ClusterInfo> _clusterInfos;
	
	
	public ClusteringKMeansModel(int k, double fuzziness) {
		_k = k;
		_clusterer = new FuzzyKMeansClusterer<Observation>(_k, fuzziness);
	}

	@Override
	public void feedData(double[] data, float label) {
		_points.add( new Observation(data, label) );
	}

	@Override
	public void build() {
		_clusters = (List<CentroidCluster<Observation>>)_clusterer.cluster(_points);
		_clusterInfos = new ArrayList<ClusteringKMeansModel.ClusterInfo>( _clusters.size() );
		for(Cluster<Observation> cluster : _clusters)
			_clusterInfos.add( makeClassInfo(cluster) );
		
		for(ClusterInfo info : _clusterInfos)
			System.out.print( String.format("%.5f", info._purity) + " ");
		System.out.println();
	}

	@Override
	public double estimate(double data[]) {
		DistanceMeasure distanceMeasure = _clusterer.getDistanceMeasure();
		double minDistance = Double.MAX_VALUE;
		int closestPointIndex = 0;

		for(int i=0; i<_points.size(); ++i) {
			double distanceFromPoint = distanceMeasure.compute(_points.get(i)._data, data);
			if(distanceFromPoint < minDistance) {
				minDistance = distanceFromPoint;
				closestPointIndex = i;
			}
		}
		
		RealMatrix membershipMatrix = _clusterer.getMembershipMatrix();
		double[] membershipWeights = membershipMatrix.getRow(closestPointIndex);
		double positiveWeights = 0;
		double negativeWeights = 0;
		for(int i=0; i<membershipWeights.length; ++i) {
			if( _clusterInfos.get(i)._label >= PairData.LABEL_PARAPHRASE06 )
				positiveWeights += membershipWeights[i];
			else
				negativeWeights += membershipWeights[i];
		}
		
		double positiveEstimation = 0;
		double negativeEstimation = 0;
		
		for(int i=0; i<membershipWeights.length; ++i) {
			if( _clusterInfos.get(i)._label >= 0.6 )
				positiveEstimation += ( (membershipWeights[i]/positiveWeights) * _clusterInfos.get(i)._label);
			else
				negativeEstimation += ( (membershipWeights[i]/negativeWeights) * _clusterInfos.get(i)._label);
		}
		
		return positiveWeights > negativeWeights ? positiveEstimation : negativeEstimation;
	}
	
	protected ClusterInfo makeClassInfo(Cluster<Observation> cluster) {
		Map<Float, Integer> classOccurrences = new HashMap<Float, Integer>();
		float label;
		
		for( Observation observation : cluster.getPoints() ) {
			label = observation.getLabel();
			int occurrences = classOccurrences.containsKey(label) ? (classOccurrences.get(label) + 1) : 0;
			classOccurrences.put(label, occurrences);
		}
		
		int maxOccurences = Integer.MIN_VALUE;
		label = 0f;
		float entropy = 0;
		
		for(Map.Entry<Float, Integer> entry : classOccurrences.entrySet()) {
			if(entry.getValue() > maxOccurences)
				maxOccurences = entry.getValue();
			label += entry.getKey() * entry.getValue() / (double)cluster.getPoints().size();
			double p = entry.getValue() / (double)cluster.getPoints().size();
			entropy -= p * log2(p);
		}
		
		return new ClusterInfo(
				label,
				maxOccurences / (double)cluster.getPoints().size(),
				entropy );
	}
	
	private double log2(double x) {
		return Math.log(x) / Math.log(2);
	}
	
	public double getPurity() {
		double purity = 0;
		for(int i=0; i<_clusters.size(); ++i)
			purity += 
				_clusterInfos.get(i)._purity
				* (_clusters.get(i).getPoints().size() / (double)_points.size());
			
		return purity;
	}
	
	public double getEntropy() {
		double entropy = 0;
		for(int i=0; i<_clusters.size(); ++i)
			entropy += _clusterInfos.get(i)._entropy
			* (_clusters.get(i).getPoints().size() / (double)_points.size());
		
		return entropy;
	}
	
	public double evaluate() {
		ClusterEvaluator<Observation> evaluator = 
				new SumOfClusterVariances<Observation>(_clusterer.getDistanceMeasure());
		
		return evaluator.score(_clusters);
	}
}
