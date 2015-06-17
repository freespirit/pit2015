package trifonov.stanislav.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.Clusterer;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.evaluation.ClusterEvaluator;
import org.apache.commons.math3.ml.clustering.evaluation.SumOfClusterVariances;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

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
	private final Clusterer<Observation> _clusterer;
	private List<CentroidCluster<Observation>> _clusters;
	private List<ClusterInfo> _clusterInfos;
	
	
	public ClusteringKMeansModel(int k) {
		_k = k;
		_clusterer = new KMeansPlusPlusClusterer<Observation>(_k);
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
		
//		for(ClusterInfo info : _clusterInfos)
//			System.out.print( String.format(Locale.US, "%.4f", info._purity) + " | ");
//		System.out.println();
	}

	@Override
	public double estimate(double data[]) {
		DistanceMeasure distanceMeasure = _clusterer.getDistanceMeasure();
		double minDistance = Double.MAX_VALUE;
		ClusterInfo info = null;
		
		for(int i=0; i<_clusters.size(); ++i) {
			CentroidCluster<Observation> cluster = _clusters.get(i);
			double distanceFromCenter = distanceMeasure.compute( cluster.getCenter().getPoint(), data );
			if(distanceFromCenter < minDistance) {
				minDistance = distanceFromCenter;
				info = _clusterInfos.get(i);
			}
		}
		
		return info._label;
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
		float dominantLabel = Float.MIN_VALUE;
		float entropy = 0;
		
		for(Map.Entry<Float, Integer> entry : classOccurrences.entrySet()) {
			if(entry.getValue() > maxOccurences) {
				maxOccurences = entry.getValue();
				dominantLabel = entry.getKey();
			}
			double p = entry.getValue() / (double)cluster.getPoints().size();
			entropy -= p * log2(p);
		}
		
		return new ClusterInfo(
				dominantLabel,
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
				new SumOfClusterVariances<ClusteringKMeansModel.Observation>(_clusterer.getDistanceMeasure());
		
		return evaluator.score(_clusters);
	}
}
