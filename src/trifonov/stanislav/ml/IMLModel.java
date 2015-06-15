package trifonov.stanislav.ml;

public interface IMLModel {

	public void feedData(double data[], float label);
//	public void setData(double data[]);
	public void build();
	public double estimate(double data[]);
}
