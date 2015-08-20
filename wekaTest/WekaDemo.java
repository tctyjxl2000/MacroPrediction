package wekaTest;

import java.io.*;
import weka.core.converters.CSVLoader;
import weka.core.converters.DatabaseLoader;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MathExpression;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.instance.RemoveFolds;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;


public class WekaDemo {

	/**
	 * @param args
	 */
	int filterStep;
	int classifierStep;
	Filter[] filter;
	Classifier[] classifier;
	
	public WekaDemo(){
		filterStep=4;
		filter = new Filter[filterStep];
		for(int ii=0;ii<filterStep;ii++){
			filter[ii] = new Reorder();
		}
	}
	
	public void load(String args) throws Exception {
		String path = "/home/zm/workspace/test/base_components";
		File dir = new File(path);
		File[] models = dir.listFiles();
		int[] indexFilter = new int[10];
		int[] indexClassifier = new int[10];
		int filterNum = 0;
		int classifierNum = 0;
		int i = 0;
		while (true){
			if (models[i].getName().startsWith("filter") & models[i].getName().endsWith((filterNum+1)+""+".model")){
				indexFilter[filterNum] = i;
				filterNum++;
				}
			else if (models[i].getName().startsWith("classifier") & models[i].getName().endsWith((classifierNum+1)+""+".model")){
				indexClassifier[classifierNum] = i;
				classifierNum++;
				}
			if(++i >= models.length){i=0;}
			if(filterNum+classifierNum >= models.length){break;}
		}
		this.filterStep = filterNum;
		this.classifierStep = classifierNum;
		this.filter = new Filter[this.filterStep];
		for (int ii=0; ii<this.filterStep;ii++){
			this.filter[ii] = (Filter) SerializationHelper.read(models[indexFilter[ii]].getAbsolutePath());
		}
		this.classifier = new Classifier[this.classifierStep];
		for(int ii=0;ii<this.classifierStep;ii++){
			this.classifier[ii] = (Classifier) SerializationHelper.read(models[indexClassifier[ii]].getAbsolutePath());
		}
	}
	
	public void classify(String args) throws Exception {
		// The following is just for test of the procedure
		String DB_dir = "jdbc:mysql://localhost:3306/traffic";
		String Query_command = "select * from t_traffic_net";
		DatabaseLoader loader = new DatabaseLoader();
		loader.setSource(DB_dir, "root", "asdf");
		loader.setQuery(Query_command);
		Instances data_raw = loader.getDataSet();
		String[] opt_filter_test = new String[6];
		opt_filter_test[0] = "-S";
		opt_filter_test[1] = "0";
		opt_filter_test[2] = "-N";
		opt_filter_test[3] = "3"; // Use 2/3 for training, another 1/3 for
									// testing
		opt_filter_test[4] = "-F";
		opt_filter_test[5] = "1";
		RemoveFolds rf = new RemoveFolds();
		rf.setOptions(opt_filter_test);
		rf.setInputFormat(data_raw);
		Instances testing = Filter.useFilter(data_raw, rf);
		System.out.println("The number of instances for testing is "
				+ testing.numInstances());

		for (int ii = 0; ii < this.filterStep; ii++) {
			testing = Filter.useFilter(testing, this.filter[ii]);
		}
		testing.setClassIndex(testing.numAttributes() - 1);
		for (int jj = 0; jj < this.filterStep; jj++) {
			System.out.println("# - actual - predicted - distribution");
			for (int ii = 0; ii < testing.numInstances(); ii++) {
				double pred = this.classifier[jj].classifyInstance(testing
						.instance(ii));
				double[] dist = this.classifier[jj]
						.distributionForInstance(testing.instance(ii));
				System.out.print((ii + 1) + " - ");
				System.out.print(testing.instance(ii).toString(
						testing.classIndex())
						+ " - ");
				System.out.print(testing.classAttribute().value((int) pred)
						+ " - "); // weka.core.attribute.value saves all the
									// possible values in the data set
				System.out.println(Utils.arrayToString(dist));
			}
		}
	}
	
	public void train(String args) throws Exception {
		// The example of loading data from files
		/*File filesource = new File("/home/zm/workspace/traffic_raw_data/data/TT_20120101/TT_201201010000.csv");
		CSVLoader loader = new CSVLoader();
		try{
			loader.setSource(filesource);
		}catch(IOException e){
			throw new Exception("File can not be found!",e);
			
		}*/
		// The example of loading data from database (MySQL. For the JDBC driver of 
		// MySQL, please refer to the doc. Weka_using)
		int filterStep = 1;
		int classifierStep = 1;
		String modelPath = "/home/zm/workspace/test/base_components/";
		String DB_dir = "jdbc:mysql://localhost:3306/traffic";
		String Query_command = "select * from t_traffic_net";
		DatabaseLoader loader = new DatabaseLoader();
		loader.setSource(DB_dir,"root","asdf");
		loader.setQuery(Query_command);
		Instances data_raw = loader.getDataSet();
		
		// The example of setting options of a filter. Here we use the Math Expression to transform the problem
		String[] opt_filter_discret = new String[5];
		opt_filter_discret[0] = "-E";
		opt_filter_discret[1] = "ifelse(A<3,0,1)";
		opt_filter_discret[2] = "-V";
		opt_filter_discret[3] = "-R";
		opt_filter_discret[4] = "1";
		Filter me = new MathExpression();
		me.setOptions(opt_filter_discret);
		
		// The example of filtering data
		me.setInputFormat(data_raw);
		Instances data_classed = Filter.useFilter(data_raw, me);
		SerializationHelper.write(modelPath+"filterStep"+(filterStep+"")+".model", me);
		filterStep++;
		//this.filter[0] = (Filter) SerializationHelper.read("/home/zm/workspace/test/base_components/filterStep1.model");
		//Instances data_ttt = Filter.useFilter(data_raw, this.filter[0]);
		//System.out.println(data_ttt.variance(0));
		//System.out.println(data_classed.variance(0));
		//System.out.println(data_raw.variance(0));
		//Filter.useFilter(data_raw, me);
		// = new MathExpression();
		
		// The reordering example (so the class label is moved to the last attribute, which is the default of WEKA)
		String[] opt_filter_reorder = new String[2];
		opt_filter_reorder[0] = "-R";
		opt_filter_reorder[1] = "2-last,1";
		Reorder ro = new Reorder();
		ro.setOptions(opt_filter_reorder);
		ro.setInputFormat(data_classed);
		Instances data_formated = Filter.useFilter(data_classed, ro);
		SerializationHelper.write(modelPath+"filterStep"+(filterStep+"")+".model", ro);
		filterStep++;
		
		// The step to transform numeric labels to nominal labels, so that typical classifier can be used
		String[] opt_filter_nominal = new String[2];
		opt_filter_nominal[0] = "-R";
		opt_filter_nominal[1] = "last";
		NumericToNominal num2nom = new NumericToNominal();
		num2nom.setOptions(opt_filter_nominal);
		num2nom.setInputFormat(data_formated);
		Instances data_nominal = Filter.useFilter(data_formated, num2nom);
		SerializationHelper.write(modelPath+"filterStep"+(filterStep+"")+".model", num2nom);
		filterStep++;
		
		// The step to separate training set & test set. If cross-validation is applied, this is not necessary
		String[] opt_filter_test = new String[6];
		opt_filter_test[0] = "-S";
		opt_filter_test[1] = "0";
		opt_filter_test[2] = "-N";
		opt_filter_test[3] = "3";  // Use 2/3 for training, another 1/3 for testing
		opt_filter_test[4] = "-F";
		opt_filter_test[5] = "1";
		RemoveFolds rf = new RemoveFolds();
		rf.setOptions(opt_filter_test);
		rf.setInputFormat(data_nominal);
		Instances testing = Filter.useFilter(data_nominal, rf);
		System.out.println("The number of instances for testing is " + testing.numInstances());
		
		String[] opt_filter_train = new String[7];
		for(int ii = 0; ii<opt_filter_test.length; ii++){
			opt_filter_train[ii] = opt_filter_test[ii];
		}
		opt_filter_train[opt_filter_train.length-1] = "-V";
		rf.setOptions(opt_filter_train);
		Instances training = Filter.useFilter(data_nominal, rf);
		
		// The step to normalize the testing and training data for SVM classifier.
		String[] opt_filter_normalize = new String[4];
		opt_filter_normalize[0] = "-S";
		opt_filter_normalize[1] = "2.0";
		opt_filter_normalize[2] = "-T";
		opt_filter_normalize[3] = "-1.0";
		Normalize nm = new Normalize();
		nm.setOptions(opt_filter_normalize);
		nm.setInputFormat(training);
		Instances newTraining = Filter.useFilter(training, nm);
		Instances newTesting = Filter.useFilter(testing, nm);
		SerializationHelper.write(modelPath+"filterStep"+(filterStep+"")+".model", nm);
		filterStep++;
		// Need to set the class index so that classifier knows which is the label.
		newTraining.setClassIndex(newTraining.numAttributes()-1);
		newTesting.setClassIndex(newTesting.numAttributes()-1);
		
		// The step to build a SVM classifier based on the training data
		String[] opt_filter_SVM = new String[2];
		opt_filter_SVM[0] = "-N";
		opt_filter_SVM[1] = "2"; // string 0-1 ensures no further normalization or standardization 
		//applied (since it's done in previous step by our own filter)
		SMO classifierSMO = new SMO();
		classifierSMO.setOptions(opt_filter_SVM);
		RBFKernel rbf = new RBFKernel();
		rbf.setGamma(0.01);
		classifierSMO.setKernel(rbf);
		
		classifierSMO.buildClassifier(newTraining);
		
		//SerializationHelper.write("/home/zm/workspace/test/base_components/classifierStep.model", classifierSMO);
		SerializationHelper.write(modelPath+"classifierStep"+(classifierStep+"")+".model", classifierSMO);
		classifierStep++;
		
		System.out.println("# - actual - predicted - distribution");
		for (int ii=0; ii<newTesting.numInstances();ii++){
			double pred = classifierSMO.classifyInstance(newTesting.instance(ii));
			double[] dist = classifierSMO.distributionForInstance(newTesting.instance(ii));
			System.out.print((ii+1) + " - ");
			System.out.print(newTesting.instance(ii).toString(newTesting.classIndex())+ " - ");
			System.out.print(newTesting.classAttribute().value((int) pred)+ " - ");  // weka.core.attribute.value saves all the possible values in the data set
			System.out.println(Utils.arrayToString(dist));
		}
		
	}
	public static void main(String[] args) throws Exception {
		WekaDemo ot = new WekaDemo();
		ot.train("-a ss");
	}
}
