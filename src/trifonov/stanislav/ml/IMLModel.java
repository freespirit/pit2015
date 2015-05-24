package trifonov.stanislav.ml;

public interface IMLModel {

	public void feedData(double data[], double label);
	public void build();
	public double estimate(double data);
}
