package trifonov.stanislav.ml;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.FuzzyKMeansClusterer;
import org.apache.commons.math3.ml.clustering.evaluation.ClusterEvaluator;
import org.apache.commons.math3.ml.clustering.evaluation.SumOfClusterVariances;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

import com.xeiam.xchart.BitmapEncoder;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Histogram;
import com.xeiam.xchart.BitmapEncoder.BitmapFormat;
import com.xeiam.xchart.StyleManager.ChartType;
import com.xeiam.xchart.StyleManager.LegendPosition;

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
	
	
	private final List<Observation> _points = new ArrayList<Observation>();
	private final DBSCANClusterer<Observation> _clusterer;
	private List<Cluster<Observation>> _clusters;
	private List<ClusterInfo> _clusterInfos;
	
	
	public ClusteringKMeansModel(int minPts, double eps) {
		_clusterer = new DBSCANClusterer<ClusteringKMeansModel.Observation>(eps, minPts);
	}

	@Override
	public void feedData(double[] data, float label) {
		_points.add( new Observation(data, label) );
	}

	@Override
	public void build() {
		///
//		File dir = new File("dbscan");
//		for(File f : dir.listFiles())
//			f.delete();
//		
//		int pointsCount = _points.size();
//		DistanceMeasure dm = _clusterer.getDistanceMeasure();
//		double[][] distMatrix = new double[pointsCount][pointsCount];
//		for(int i=0; i<pointsCount; ++i) {
//			double[] distances = distMatrix[i];
//			for(int j=0; j<pointsCount; ++j) {
//				distances[j] = dm.compute(_points.get(i)._data, _points.get(j)._data);
//			}
//			Arrays.sort(distances);
//		}
//		
//		for(int k=2; k<pointsCount-1; ++k) {
//			List<Double> kDistances = new ArrayList<Double>(pointsCount);
//			
//			for(int i=0; i<pointsCount; ++i)
//				kDistances.add( new Double(distMatrix[i][k+1]) );			
//			
//
//			Collections.sort(kDistances);
//			List<Double> yData = new ArrayList<>();
//			List<Integer> xData = new ArrayList<>();
//			for(int i=0; i<1000; ++i) {
//				yData.add( kDistances.get(i) );
//				xData.add( i );
//			}
////			Histogram hist = new Histogram(kDistances, pointsCount);
//			Chart chart = new ChartBuilder().chartType(ChartType.Line)
//					.width(1024)
//					.height(768)
//					.title(k+"-distance")
//					.xAxisTitle("k")
//					.yAxisTitle("eps")
//					.build();
//			
//			chart.addSeries("K-distances", xData, yData);
//			chart.getStyleManager().setLegendPosition(LegendPosition.InsideNE);
//			try {
//				dir.mkdirs();
//				BitmapEncoder.saveBitmap(chart, new File(dir, k + "-distance").getAbsolutePath(), BitmapFormat.PNG);
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//		}
		///
		
		
		
		_clusters = _clusterer.cluster(_points);
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

		for(int i=0; i<_clusters.size(); ++i) {
			for( Observation observation : _clusters.get(i).getPoints() ) {
				if( distanceMeasure.compute(data, observation._data) < _clusterer.getEps() )
					return _clusterInfos.get(i)._label;
			}
		}
		
		return PairData.LABEL_DEBATABLE;
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
		label = 0f;
		float entropy = 0;
		
		for(Map.Entry<Float, Integer> entry : classOccurrences.entrySet()) {
			if(entry.getValue() > maxOccurences) {
				maxOccurences = entry.getValue();
				dominantLabel = entry.getKey();
			}
			label += entry.getKey() * entry.getValue() / (double)cluster.getPoints().size();
			double p = entry.getValue() / (double)cluster.getPoints().size();
			entropy -= p * log2(p);
		}
		
		return new ClusterInfo(
//				dominantLabel,
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
