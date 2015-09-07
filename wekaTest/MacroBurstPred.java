package wekaTest;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.converters.CSVLoader;
import weka.core.Attribute;
import weka.core.Instances;
import weka.attributeSelection.AttributeSelection;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.Instance;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.MathExpression;
//import weka.filters.supervised.attribute.AttributeSelection;
import weka.classifiers.trees.RandomForest;
import weka.core.AttributeStats;
import weka.filters.unsupervised.attribute.Normalize;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.clusterers.ClusterEvaluation;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;








//import org.apache.commons.math3.stat.*;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class MacroBurstPred {
	int filterstep;
	Instances data_nominal;
	Instances data_full;
	Instances data_test;
	boolean cluster_first;
	Filter[] filter;
	RandomForest classifier;
	SimpleKMeans clusterer;
	String type;
	AttributeSelection[] att_sel;
	double resample_ratio;
	
	public MacroBurstPred(){
		filterstep = 3;
		filter = new Filter[filterstep];		
		att_sel = new AttributeSelection[5];
//		classifier = new RandomForest();
	}
	
	public void dataLoad() throws Exception{
		String data_file_name = "./data_source/training_set_成都_2G互联网_有信号但无法使用_thre=13_randomized_periodic_1hour.csv";
		//String data_file_name = "./data_source/test.csv";
//		String data_file_name = "./data_source/training_set_CS_coverage_thre=8_randomized.csv";
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(data_file_name));
		Instances data_raw = loader.getDataSet();
		String[] opt_filter_nominal = new String[2];
		opt_filter_nominal[0] = "-R";
//		opt_filter_nominal[1] = "last";
		opt_filter_nominal[1] = String.valueOf(data_raw.attribute("突发投诉").index()+1); // Note: Here the index shall be the real one plus 1, since in the input format, index starts from 1 instead of 0
		NumericToNominal num2nom = new NumericToNominal();
		num2nom.setOptions(opt_filter_nominal);
		num2nom.setInputFormat(data_raw);
		Instances data_nominal = Filter.useFilter(data_raw, num2nom);
		this.data_full = data_nominal;
	}
	
	public void setResampleRatio(double resampling_ratio)
	{
		try{
			((Resample) this.filter[0]).setBiasToUniformClass(resampling_ratio);
		}
		catch(Exception e){
			System.out.println("The filter is not initialized yet! You are not supposed to set up the resampling ratio here.");
		}
	}
	
	public SimpleKMeans clusteringGen(Instances data_input, int numCluster) throws Exception
	{
		Remove filter = new Remove();
		filter.setAttributeIndices("" + (data_input.classIndex() + 1));
		filter.setInputFormat(data_input);
		Instances data_no_label = Filter.useFilter(data_input, filter);
		
		String kmeansconfig = String.format("-init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -N %d -I 500 -num-slots 1 -S 10", numCluster);
		String[] str_opt_kmeans = kmeansconfig.split(" ");
		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setOptions(str_opt_kmeans);
		
		EuclideanDistance EDist = new EuclideanDistance();
		EDist.setDontNormalize(true);
//		EDist.setAttributeIndices(String.format("1-%d", data_input.numAttributes()));
		kmeans.setDistanceFunction(EDist);
		kmeans.buildClusterer(data_no_label);
		return kmeans;
	}
	
	public double[] one_pass_RF(Instances train, Instances test, boolean reweight, boolean resample) throws Exception{
		double[] result = new double[19];
		String file_name = "./data_source/forest_example.dat";
		FileWriter fw = new FileWriter(file_name,true);
		if (reweight){
			this.filter[1].setInputFormat(train);
			train = Filter.useFilter(train, this.filter[1]);
		}
		if (resample){
			this.filter[0].setInputFormat(train);
			train = Filter.useFilter(train, this.filter[0]);
		}
		AttributeStats stat = train.attributeStats(train.classIndex());
		System.out.println(Arrays.toString(stat.nominalCounts));
		System.out.println(Arrays.toString(stat.nominalWeights));
		
		Evaluation eval = new Evaluation(train);
//		classifierJ48.buildClassifier(train);			
//		eval.evaluateModel(classifierJ48, test);
		this.classifier.buildClassifier(train);
//		fw.write(this.classifier.toString());
//		fw.close();
		// The following is only used for testing of method getMembershipValues
//		Instance aa = test.instance(1);
//		double[] bb = this.classifier.getMembershipValues(aa);
//		System.out.println(Arrays.toString(bb));
		eval.evaluateModel(this.classifier, test);
		
		double[][] tmp = eval.confusionMatrix();
		for (int index_1 = 0; index_1 < tmp.length;index_1++){
			for (int index_2 = 0; index_2 < tmp[index_1].length; index_2++){
				result[15+index_1*2+index_2] += tmp[index_1][index_2];
			}
		}
//		fw.write(eval.toSummaryString("\nResults\n\n",false));
//		fw.write(eval.toMatrixString("\nResults\n\n"));
//		fw.write(eval.toClassDetailsString());
		for (int classIndex=0; classIndex<=1; classIndex+=1){
			result[classIndex*5+0] += eval.truePositiveRate(classIndex);
			result[classIndex*5+1] += eval.falsePositiveRate(classIndex);
			result[classIndex*5+2] += eval.precision(classIndex);
			result[classIndex*5+3] += eval.recall(classIndex);
			result[classIndex*5+4] += eval.fMeasure(classIndex);
		}
		result[10] += eval.weightedTruePositiveRate();
		result[11] += eval.weightedFalsePositiveRate();
		result[12] += eval.weightedPrecision();
		result[13] += eval.weightedRecall();
		result[14] += eval.weightedFMeasure();
		return result;
	}
	
	public double[] train_RF(int folds, int maxdepth, int features, int numTrees, boolean resample, boolean reweight, double resample_ratio) throws Exception{
		String file_result = "./data_source/result_compare.dat";
		FileWriter fw = new FileWriter(file_result, true);
		
		Instances data_nominal = this.data_nominal;
		double[] result = new double[19];
		String[] opt_filter_Resample = new String[]{"-B","1.0","-S","1","-Z","100.0"};
//		opt_filter_Resample[1] = String.valueOf(resample_ratio);
		Resample balancer = new Resample();
		balancer.setOptions(opt_filter_Resample);		
		this.filter[0] = balancer;
		if (resample) {
			fw.write("Resample Filter applied: ");
			fw.write( Arrays.toString(opt_filter_Resample)+"\n");
			fw.write(folds+" folds; \n");
		}
		this.setResampleRatio(resample_ratio);  // Set the resampling ratio to the input one
		ClassBalancer weight_balancer = new ClassBalancer();
		this.filter[1] = weight_balancer;
		if (reweight){
			fw.write("ClassBalancer Filter applied with default configuration\n");
		}
		String[] str_opt_J48 = new String[4];
		str_opt_J48[0] = "-C";
		str_opt_J48[1] = "0.2";
		str_opt_J48[2] = "-M";
		str_opt_J48[3] = "5";	
		J48 classifierJ48 = new J48();
//		fw.write("J48 dicision tree applied "+ Arrays.toString(str_opt_J48)+"\n");
		classifierJ48.setOptions(str_opt_J48);
		
		String[] str_opt_RandomForest = new String[]{"-I",Integer.toString(numTrees),"-K",Integer.toString(features),"-depth",Integer.toString(maxdepth),"-num-slots","2"};
		RandomForest classifierRF = new RandomForest();
		fw.write("RandomForest algorithm applied" + Arrays.toString(str_opt_RandomForest)+"\n");
		classifierRF.setOptions(str_opt_RandomForest);
		classifierRF.setPrintTrees(true);
		this.classifier = classifierRF;
		double confusion_mat[][] = {{0,0},{0,0}};
		int prec_effective_count = 0;  // This is to count how many precision values returned in the multi-folds test is effective, 
		// since in some folds, the TP and FP are both 0. The precision is therefore meaningless.
		//Arrays.fill(confusion_mat, 0);
		for (int ii = 0; ii < folds; ii++){
			Instances train = data_nominal.trainCV(folds, ii);
			Instances test = data_nominal.testCV(folds, ii);
			train.setClassIndex(train.attribute("突发投诉").index());
			test.setClassIndex(test.attribute("突发投诉").index());
			
			if (this.type=="Feature_Selected"){
				int[] att_index = this.att_sel[ii].selectedAttributes();
				System.out.println("Selected feature length is " + att_index.length);
				Remove rm = new Remove();
				rm.setAttributeIndicesArray(att_index);
				rm.setInvertSelection(true);
				rm.setInputFormat(train);
				train = Filter.useFilter(train, rm);
				test = Filter.useFilter(test, rm);
			}
			
			double[] result_fold = this.one_pass_RF(train, test, reweight, resample);
			
			for (int index_1 = 0; index_1 < confusion_mat.length;index_1++){
				for (int index_2 = 0; index_2 < confusion_mat[index_1].length; index_2++){
					confusion_mat[index_1][index_2] +=result_fold[15+2*index_1+index_2];
				}
			}
			
			// structure of result[]: index 0-4: TP, FP, precision, recall, F Measure of non-bursting instances; index 5-9: similar metrics of bursting instances.
			// index: 10-14: similar metrics, but weighted among 2 different classes. index 15-18: confusion matrix
			prec_effective_count++;
			System.out.println(result_fold[7]);
			if ((result_fold[7]==0) && (confusion_mat[0][1]==0)){
				prec_effective_count--;
				System.out.println("One meaningless precision is calculated");
			}
			for (int stats = 0; stats<15; stats++){
				result[stats]+=result_fold[stats];
			}
		}
		fw.write(Arrays.toString(confusion_mat[0]));
		fw.write(Arrays.toString(confusion_mat[1])+"\n");
		for (int ii = 0; ii<15; ii++){
			if (ii!=7){
				result[ii] = result[ii]/folds;}
			else{
				result[ii]=(prec_effective_count==0)?0.0:result[ii]/prec_effective_count;}
		}
		for (int index_1 = 0; index_1 < confusion_mat.length;index_1++){
			for (int index_2 = 0; index_2 < confusion_mat[index_1].length; index_2++){
				result[15+index_1*2+index_2]=confusion_mat[index_1][index_2];
			}
		}
		
		fw.close();
		return result;
	}
	
	public int[] classOfCluster(SimpleKMeans kmeans, Instances train_norm) throws Exception{
		ClusterEvaluation eval_c = new ClusterEvaluation();
		eval_c.setClusterer(kmeans);
		eval_c.evaluateClusterer(train_norm);
		double[] clusters = eval_c.getClusterAssignments();
		double[] labels = train_norm.attributeToDoubleArray(train_norm.classIndex());
		int[][] classesToClusters = new int[eval_c.getNumClusters()][2];
		for(int jj=0; jj<clusters.length;jj++){
			classesToClusters[(int)clusters[jj]][(int)labels[jj]] += 1;
		}
		int[] classOfCluster = new int[classesToClusters.length];
		for (int jj=0;jj<classesToClusters.length;jj++){
//			System.out.println(Arrays.toString(classesToClusters[jj]));
			if (classesToClusters[jj][0] == 0){classOfCluster[jj]=1;}
			else if  (classesToClusters[jj][1] == 0){classOfCluster[jj]=0;}
			else if (classesToClusters[jj][0]/classesToClusters[jj][1]>15){classOfCluster[jj]=0;}
			else if (classesToClusters[jj][1]/classesToClusters[jj][0]>15){classOfCluster[jj]=1;}
			else{classOfCluster[jj]=-1;}
		}
		return classOfCluster;
	}
	
	public List clusterFiltering(SimpleKMeans kmeans, Instances train_norm, int[] classOfCluster, Instances train) throws Exception{
		double[][] confusion_mat_cluster = {{0,0},{0,0}};
		ClusterEvaluation eval_c = new ClusterEvaluation();
		eval_c.setClusterer(kmeans);
		eval_c.evaluateClusterer(train_norm);
		double[] clusters = eval_c.getClusterAssignments();
		double[] labels = train_norm.attributeToDoubleArray(train_norm.classIndex());
		Instances train_tmp = new Instances(train_norm,0);
		for (int jj=0;jj<train_norm.numInstances();jj++){
			int cluster_label = (int)clusters[jj];
			if (classOfCluster[cluster_label]==-1){	train_tmp.add(train.instance(jj));}
			else{confusion_mat_cluster[(int)labels[jj]][classOfCluster[cluster_label]]+=1;}
		}
		List result = new ArrayList();
		result.add(train_tmp);
		result.add( confusion_mat_cluster);
		return result;
	}
	
	public double[] one_pass_RF_cluster(Instances train, Instances test,  boolean reweight, boolean resample, int numClusters) throws Exception{
		this.filter[2].setInputFormat(train);
//		double[][] confusion_mat = {{0,0},{0,0}};
//		double[] result = new double[19];
		Instances train_norm = Filter.useFilter(train, this.filter[2]);
		Instances test_norm = Filter.useFilter(test, this.filter[2]);
//		Instances train_norm = data_normalized.trainCV(folds, ii);
//		Instances test_norm = data_normalized.testCV(folds, ii);
//		train_norm.setClassIndex(train_norm..attribute("突发投诉").index());
//		test_norm.setClassIndex(test_norm..attribute("突发投诉").index());
		SimpleKMeans kmeans = this.clusteringGen(train_norm, numClusters);
		this.clusterer = kmeans;
		
		int[] classOfCluster = this.classOfCluster(kmeans, train_norm);
		
		List tmp_return = this.clusterFiltering(kmeans, train_norm, classOfCluster,train);
		train = (Instances) tmp_return.get(0);
		System.out.println(train.numInstances());
		
		if (reweight){
			this.filter[1].setInputFormat(train);
			train = Filter.useFilter(train, this.filter[1]);
		}
		if (resample){
			this.filter[0].setInputFormat(train);
			train = Filter.useFilter(train, this.filter[0]);
		}
//		AttributeStats stat = train.attributeStats(train.classIndex());
//		System.out.println(Arrays.toString(stat.nominalCounts));
//		System.out.println(Arrays.toString(stat.nominalWeights));
		
		Evaluation eval = new Evaluation(train);
//		classifierJ48.buildClassifier(train);			
//		eval.evaluateModel(classifierJ48, test);			
		this.classifier.buildClassifier(train);			
		
		tmp_return = this.clusterFiltering(kmeans, test_norm, classOfCluster, test);
		test = (Instances) tmp_return.get(0);
		double[][] confusion_mat_cluster = (double[][]) tmp_return.get(1);
		
		System.out.println(test.numInstances());
		System.out.println(Arrays.toString(confusion_mat_cluster[0]));
		System.out.println(Arrays.toString(confusion_mat_cluster[1]));
		eval.evaluateModel(this.classifier, test);
		
		double[][] tmp = eval.confusionMatrix();
		double[] result_fold = new double[19];
		for (int index_1 = 0; index_1 < tmp.length;index_1++){
			for (int index_2 = 0; index_2 < tmp[index_1].length; index_2++){
				tmp[index_1][index_2] += confusion_mat_cluster[index_1][index_2];
				result_fold[15+index_1*2+index_2] += tmp[index_1][index_2];
			}
		}
//		System.out.println(Arrays.toString(tmp[0]));
//		System.out.println(Arrays.toString(tmp[1]));
//		fw.write(eval.toSummaryString("\nResults\n\n",false));
//		fw.write(eval.toMatrixString("\nResults\n\n"));
//		fw.write(eval.toClassDetailsString());
		double negative_ratio = (double)(tmp[0][0]+tmp[0][1])/(tmp[0][0]+tmp[0][1]+tmp[1][0]+tmp[1][1]);
		for (int classIndex=0; classIndex<=1; classIndex+=1){
			result_fold[classIndex*5+0] = (double)tmp[classIndex][classIndex]/(tmp[classIndex][classIndex]+tmp[classIndex][1-classIndex]);
//			result[classIndex*5+0] += result_fold[classIndex*5+0];
			result_fold[classIndex*5+1] = (double)tmp[1-classIndex][classIndex]/(tmp[1-classIndex][classIndex]+tmp[1-classIndex][1-classIndex]);
//			result[classIndex*5+1] += result_fold[classIndex*5+1];
			if (tmp[classIndex][classIndex]+tmp[1-classIndex][classIndex]>0){
				result_fold[classIndex*5+2] = (double)tmp[classIndex][classIndex]/(tmp[classIndex][classIndex]+tmp[1-classIndex][classIndex]);}
//				result[classIndex*5+2] += result_fold[classIndex*5+2];}
			result_fold[classIndex*5+3] = result_fold[classIndex*5+0];
//			result[classIndex*5+3] += result_fold[classIndex*5+3];
			if ((result_fold[classIndex*5+2]+result_fold[classIndex*5+3])!=0){
				result_fold[classIndex*5+4] = 2*result_fold[classIndex*5+2]*result_fold[classIndex*5+3]/(result_fold[classIndex*5+2]+result_fold[classIndex*5+3]);}
//				result[classIndex*5+4] += result_fold[classIndex*5+4];}
		}
		result_fold[10] += negative_ratio*result_fold[0]+(1-negative_ratio)*result_fold[5];
		result_fold[11] += negative_ratio*result_fold[1]+(1-negative_ratio)*result_fold[6];
		result_fold[12] += negative_ratio*result_fold[2]+(1-negative_ratio)*result_fold[7];
		result_fold[13] += negative_ratio*result_fold[3]+(1-negative_ratio)*result_fold[8];
		result_fold[14] += negative_ratio*result_fold[4]+(1-negative_ratio)*result_fold[9];
//		System.out.println(Arrays.toString(result));
		return result_fold;
	}
	
	public double[] train_RF_clustered(int folds, int maxdepth, int features, int numTrees, boolean resample, boolean reweight, int numClusters) throws Exception{
		String file_result = "./data_source/result_compare.dat";
		FileWriter fw = new FileWriter(file_result, true);
		
		Instances data_nominal = this.data_nominal;
		double[] result = new double[19];
		String[] opt_filter_Resample = new String[]{"-B","1.0","-S","1","-Z","100.0"};
		Resample balancer = new Resample();
		balancer.setOptions(opt_filter_Resample);		
		this.filter[0] = balancer;
		if (resample) {
			fw.write("Resample Filter applied: ");
			fw.write( Arrays.toString(opt_filter_Resample)+"\n");
			fw.write(folds+" folds; \n");
		}
		ClassBalancer weight_balancer = new ClassBalancer();
		this.filter[1] = weight_balancer;
		if (reweight){
			fw.write("ClassBalancer Filter applied with default configuration\n");
		}
		String[] str_opt_J48 = new String[4];
		str_opt_J48[0] = "-C";
		str_opt_J48[1] = "0.2";
		str_opt_J48[2] = "-M";
		str_opt_J48[3] = "5";	
		J48 classifierJ48 = new J48();
//		fw.write("J48 dicision tree applied "+ Arrays.toString(str_opt_J48)+"\n");
		classifierJ48.setOptions(str_opt_J48);
		
		String[] str_opt_RandomForest = new String[]{"-I",Integer.toString(numTrees),"-K",Integer.toString(features),"-depth",Integer.toString(maxdepth),"-num-slots","2"};
		RandomForest classifierRF = new RandomForest();
		this.classifier = classifierRF;
		fw.write("RandomForest algorithm applied" + Arrays.toString(str_opt_RandomForest)+"\n");
		classifierRF.setOptions(str_opt_RandomForest);
		
		double confusion_mat[][] = {{0,0},{0,0}};		

		String[] opt_normalize = new String[]{"-S","1.0","-T","0.0","-unset-class-temporarily"};
		Normalize normalizer = new Normalize();
		normalizer.setOptions(opt_normalize);
		this.filter[2] = normalizer;
//		normalizer.setInputFormat(data_nominal);
//		Instances data_normalized = Filter.useFilter(data_nominal, normalizer);	
		
		//Arrays.fill(confusion_mat, 0);
		for (int ii = 0; ii < folds; ii++){
			Instances train = data_nominal.trainCV(folds, ii);
			Instances test = data_nominal.testCV(folds, ii);
			train.setClassIndex(train.attribute("突发投诉").index());
			test.setClassIndex(test.attribute("突发投诉").index());
			
			double[] result_fold = this.one_pass_RF_cluster(train, test, reweight, resample, numClusters);
			
			for (int index_1 = 0; index_1 < confusion_mat.length;index_1++){
				for (int index_2 = 0; index_2 < confusion_mat[index_1].length; index_2++){
					confusion_mat[index_1][index_2] +=result_fold[15+2*index_1+index_2];
				}
			}
			
			for (int stats = 0;stats<15;stats++){
//				result_fold[classIndex*5+0] = (double)tmp[classIndex][classIndex]/(tmp[classIndex][classIndex]+tmp[classIndex][1-classIndex]);
				result[stats] += result_fold[stats];
			}
		}
		fw.write(Arrays.toString(confusion_mat[0]));
		fw.write(Arrays.toString(confusion_mat[1])+"\n");
		for (int ii = 0; ii<15; ii++){
			result[ii] = result[ii]/folds;
		}
		for (int index_1 = 0; index_1 < confusion_mat.length;index_1++){
			for (int index_2 = 0; index_2 < confusion_mat[index_1].length; index_2++){
				result[15+index_1*2+index_2]=confusion_mat[index_1][index_2];
			}
		}
		
		fw.close();
		return result;
	}
	
	public double[] gridSearch_RF(int folds) throws Exception {
		String file_result = "./data_source/result_grid_search.dat";
		int[] TreeNumber = {300,400,500};       // Set the searching range of number of trees
		int[] FeatureNumber = {18,23,28,33};	// Set the searching range of number of features for generating a tree
		int[] DepthNumber = {5,6,7};				// Set the maximum depth of the forest
		double[] FilterCombination ={0.2,0.4,0.6,0.7,0.8, 0.9,1.0};   // Set the searching range of resample ratio
//		int[] FilterCombination = {1, 2};   // To search if the resample or reweight shall be applied or not
		double[][] result_collection = new double[TreeNumber.length*FeatureNumber.length*DepthNumber.length*FilterCombination.length][19];
		for (int numTrees_index = 0; numTrees_index < TreeNumber.length; numTrees_index++){
			int numTrees = TreeNumber[numTrees_index];
			for (int  numFeature_index= 0; numFeature_index < FeatureNumber.length; numFeature_index++){
				int numFeature = FeatureNumber[numFeature_index];
				for (int depth_index = 0; depth_index < DepthNumber.length; depth_index++){
					int depth = DepthNumber[depth_index];
					for (int filtercase_index = 0; filtercase_index<FilterCombination.length; filtercase_index++){
//						int filtercase = FilterCombination[filtercase_index];
						double resample_ratio =  FilterCombination[filtercase_index];
//						le_ratio = 1.0;   // Only used for testing
						FileWriter fw = new FileWriter(file_result, true);
//						boolean resample_flag = (filtercase%2 == 1);
//						boolean reweight_flag = (filtercase/2==1);
						boolean resample_flag = true;
						boolean reweight_flag =false ;//true;
						// NOTE!!!! If the resample or reweight flag is modified, keep in mind to modify similar flags in function this.test_RF(), too.
						int numClusters = 10;
						fw.write(String.format("Instances into %d clusters\n",numClusters));
						double[] result = this.train_RF(folds, depth, numFeature, numTrees, resample_flag, reweight_flag, resample_ratio);
						result_collection[numTrees_index*(FeatureNumber.length*DepthNumber.length*FilterCombination.length)+numFeature_index*(DepthNumber.length*FilterCombination.length)+depth_index*FilterCombination.length+filtercase_index] = result;
						if (reweight_flag) {fw.write("Classes weighted based the number\t");}
						if (resample_flag) {fw.write("Classes uniformly resampled\t");}
						fw.write(String.format("Number of trees: %d, Number of features for each tree: %d, max depth: %d.\n",numTrees, numFeature, depth));
						fw.write("TP\tFP\tPrecision\tRecall\tFMeasure\n");
						fw.write(Arrays.toString(Arrays.copyOfRange(result, 0, 5))+"\tClass 0\n");
						fw.write(Arrays.toString(Arrays.copyOfRange(result, 5, 10))+"\tClass 1\n");
						fw.write(Arrays.toString(Arrays.copyOfRange(result, 10, 15))+"\tWeighted\n");
						fw.write("confusion matrix:\n");
						fw.write("Class 0, Class 1 \n");
						fw.write(Arrays.toString(Arrays.copyOfRange(result, 15, 17))+"\tClass 0\n");
						fw.write(Arrays.toString(Arrays.copyOfRange(result, 17, 19))+"\tClass 1\n");
						fw.close();
					}
				}
			}
		}
//		int target_index = 9;  // F Measure of the minor class is set as the target value
		int target_index = 7; // Precision of the minority class is selected as the objective metric
		double max = 0;
		int max_index = 0;
		for (int ii = 0; ii < result_collection.length; ii++){
			if (result_collection[ii][target_index]>max){ 
				max =result_collection[ii][target_index];
				max_index = ii;
			}
		}
		double[] parameter = new double[4];
		parameter[3] = FilterCombination[max_index%FilterCombination.length];
		max_index = max_index/FilterCombination.length;
		parameter[2] = DepthNumber[max_index%DepthNumber.length];
		max_index = max_index/DepthNumber.length;
		parameter[1] = FeatureNumber[max_index%FeatureNumber.length];
		max_index = max_index/FeatureNumber.length;
		parameter[0] = TreeNumber[max_index];		
		System.out.println(Arrays.toString(parameter));
		return parameter;
	}
	
	public int[] gridSearch_RF_clustered(int folds) throws Exception {
		String file_result = "./data_source/result_grid_search_cluster.dat";
		int[] ClusterSize = {10};
		int[] TreeNumber = {250};
		int[] FeatureNumber = {23};
		int[] DepthNumber = {4,5,6,7};
		int[] FilterCombination = {3};
		double[][] result_collection = new double[ClusterSize.length*TreeNumber.length*FeatureNumber.length*DepthNumber.length*FilterCombination.length][19];
		for (int cluster_index = 0; cluster_index<ClusterSize.length; cluster_index++){
			int numClusters = ClusterSize[cluster_index];
			for (int numTrees_index = 0; numTrees_index < TreeNumber.length; numTrees_index++){
				int numTrees = TreeNumber[numTrees_index];
				for (int  numFeature_index= 0; numFeature_index < FeatureNumber.length; numFeature_index++){
					int numFeature = FeatureNumber[numFeature_index];
					for (int depth_index = 0; depth_index < DepthNumber.length; depth_index++){
						int depth = DepthNumber[depth_index];
						for (int filtercase_index = 0; filtercase_index<FilterCombination.length; filtercase_index++){
							int filtercase = FilterCombination[filtercase_index];
							FileWriter fw = new FileWriter(file_result, true);
							boolean resample_flag = (filtercase%2 == 1);
							boolean reweight_flag = (filtercase/2==1);
//							int numClusters = 10;
							fw.write(String.format("Instances into %d clusters\n",numClusters));
							double[] result = this.train_RF_clustered(folds, depth, numFeature, numTrees, resample_flag, reweight_flag, numClusters);
							result_collection[cluster_index*(TreeNumber.length*FeatureNumber.length*DepthNumber.length*FilterCombination.length)+numTrees_index*(FeatureNumber.length*DepthNumber.length*FilterCombination.length)+numFeature_index*(DepthNumber.length*FilterCombination.length)+depth_index*FilterCombination.length+filtercase_index] = result;
							if (reweight_flag) {fw.write("Classes weighted based the number\t");}
							if (resample_flag) {fw.write("Classes uniformly resampled\t");}
							fw.write(String.format("Number of trees: %d, Number of features for each tree: %d, max depth: %d.\n",numTrees, numFeature, depth));
							fw.write("TP\tFP\tPrecision\tRecall\tFMeasure\n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 0, 5))+"\tClass 0\n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 5, 10))+"\tClass 1\n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 10, 15))+"\tWeighted\n");
							fw.write("confusion matrix:\n");
							fw.write("Class 0, Class 1 \n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 15, 17))+"\tClass 0\n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 17, 19))+"\tClass 1\n");
							fw.close();
						}
					}
				}
			}
		}
		int target_index = 9;  // F Measure of the minor class is set as the target value
		double max = 0;
		int max_index = 0;
		for (int ii = 0; ii < result_collection.length; ii++){
			if (result_collection[ii][target_index]>max){ 
				max =result_collection[ii][target_index];
				max_index = ii;
			}
		}
		int[] parameter = new int[5];
		parameter[3] = FilterCombination[max_index%FilterCombination.length];
		max_index = max_index/FilterCombination.length;
		parameter[2] = DepthNumber[max_index%DepthNumber.length];
		max_index = max_index/DepthNumber.length;
		parameter[1] = FeatureNumber[max_index%FeatureNumber.length];
		max_index = max_index/FeatureNumber.length;
		parameter[0] = TreeNumber[max_index%TreeNumber.length];
		max_index = max_index/TreeNumber.length;
		parameter[4] = ClusterSize[max_index];
		System.out.println(Arrays.toString(parameter));
		return parameter;
	}
	
	public double[] gridSearch_RF_FSelected(int folds) throws Exception {
		String file_result = "./data_source/result_grid_search_FS.dat";
		this.type = "Feature_Selected";
		double[] ClusterSize = { 0.005};//,0.0025};         // We keep the name here just to simplify the modification
		int[] TreeNumber = {200};//{200,250,300,350};
		int[] FeatureNumber = {23};//{15,20,25,30};
		int[] DepthNumber = {5,6};//{4,5,6,7};
		double[] FilterCombination = { 0.5, 0.9,1.0};//, 0.2, 0.3};
		double[][] result_collection = new double[ClusterSize.length*TreeNumber.length*FeatureNumber.length*DepthNumber.length*FilterCombination.length][19];
		for (int cluster_index = 0; cluster_index<ClusterSize.length; cluster_index++){
			double numClusters = ClusterSize[cluster_index];
			GainRatioAttributeEval grae = new GainRatioAttributeEval();
			Ranker ranker = new Ranker();
			if (numClusters <= 1) {ranker.setThreshold(numClusters);}
			else {ranker.setNumToSelect((int)numClusters);}
			for (int ii = 0; ii <folds; ii++){
				AttributeSelection as = new AttributeSelection();				
				as.setEvaluator(grae);
				as.setSearch(ranker);
				Instances train = this.data_nominal.trainCV(folds, ii);
				train.setClassIndex(train.attribute("突发投诉").index());
				Normalize normalizer = new Normalize();
				normalizer.setIgnoreClass(true);
				normalizer.setInputFormat(train);
				train = Filter.useFilter(train, normalizer);
				as.SelectAttributes(train);
				this.att_sel[ii] = as;
				System.out.println(as.selectedAttributes().length);
			}
			for (int numTrees_index = 0; numTrees_index < TreeNumber.length; numTrees_index++){
				int numTrees = TreeNumber[numTrees_index];
				for (int  numFeature_index= 0; numFeature_index < FeatureNumber.length; numFeature_index++){
					int numFeature = FeatureNumber[numFeature_index];
					for (int depth_index = 0; depth_index < DepthNumber.length; depth_index++){
						int depth = DepthNumber[depth_index];
						for (int filtercase_index = 0; filtercase_index<FilterCombination.length; filtercase_index++){
//							int filtercase = (int) FilterCombination[filtercase_index];
							FileWriter fw = new FileWriter(file_result, true);
//							boolean resample_flag = (filtercase%2 == 1);
//							boolean reweight_flag = (filtercase/2==1);
							boolean resample_flag = true;
							boolean reweight_flag = false;
							double resample_ratio = FilterCombination[filtercase_index];   // Used to define the resample ratio if theresampleing filter is selected
//							int numClusters = 10;
							if (numClusters<=1) {fw.write(String.format("Feature selection threshold is %f\n",numClusters));}
							else {fw.write(String.format("%d features shall be selected\n",(int)numClusters));}
							double[] result = this.train_RF(folds, depth, numFeature, numTrees, resample_flag, reweight_flag, resample_ratio);
							result_collection[cluster_index*(TreeNumber.length*FeatureNumber.length*DepthNumber.length*FilterCombination.length)+numTrees_index*(FeatureNumber.length*DepthNumber.length*FilterCombination.length)+numFeature_index*(DepthNumber.length*FilterCombination.length)+depth_index*FilterCombination.length+filtercase_index] = result;
							if (reweight_flag) {fw.write("Classes weighted based the number\t");}
							if (resample_flag) {fw.write("Classes uniformly resampled\t");}
							fw.write(String.format("Number of trees: %d, Number of features for each tree: %d, max depth: %d.\n",numTrees, numFeature, depth));
							fw.write("TP\tFP\tPrecision\tRecall\tFMeasure\n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 0, 5))+"\tClass 0\n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 5, 10))+"\tClass 1\n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 10, 15))+"\tWeighted\n");
							fw.write("confusion matrix:\n");
							fw.write("Class 0, Class 1 \n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 15, 17))+"\tClass 0\n");
							fw.write(Arrays.toString(Arrays.copyOfRange(result, 17, 19))+"\tClass 1\n");
							fw.close();
						}
					}
				}
			}
		}
//		int target_index = 9;  // F Measure of the minor class is set as the target value
		int target_index = 8; // Precision of the minority class is selected as the objective metric		
		double max = 0;
		double recall_thre = 0.5;
		int max_index = 0;
		for (int ii = 0; ii < result_collection.length; ii++){
			if (result_collection[ii][target_index]>max){ // && result_collection[ii][target_index+1]>=recall_thre){ 
				max =result_collection[ii][target_index];
				max_index = ii;
			}
		}
		double[] parameter = new double[5];
		parameter[3] = FilterCombination[max_index%FilterCombination.length];
		max_index = max_index/FilterCombination.length;
		parameter[2] = DepthNumber[max_index%DepthNumber.length];
		max_index = max_index/DepthNumber.length;
		parameter[1] = FeatureNumber[max_index%FeatureNumber.length];
		max_index = max_index/FeatureNumber.length;
		parameter[0] = TreeNumber[max_index%TreeNumber.length];
		max_index = max_index/TreeNumber.length;
		parameter[4] = ClusterSize[max_index];
		System.out.println(Arrays.toString(parameter));
		return parameter;
	}
	
	public void test_RF(int folds) throws Exception{
		String file_name = "./data_source/result_test.dat";
		double[][] result = new double[folds][19];
		for (int ii = 0; ii<folds; ii++){
			FileWriter fw = new FileWriter(file_name, true);
			this.data_nominal = this.data_full.trainCV(folds, ii);
			this.data_test = this.data_full.testCV(folds,ii);
			double[] config = this.gridSearch_RF(folds-1);
			fw.write(Arrays.toString(config)+"\n");
			Instances train = this.data_nominal;
			Instances test = this.data_test;
			train.setClassIndex(train.attribute("突发投诉").index());
			test.setClassIndex(test.attribute("突发投诉").index());
//			boolean resample = (config[3]%2 == 1);
//			boolean reweight = (config[3]/2==1);
			boolean resample = true;
			boolean reweight =false;
			// NOTE!!!! If the resample or reweight flag is modified, keep in mind to modify similar flags in function gridSearch_RF(), too.
			if (reweight){
				this.filter[1].setInputFormat(train);
				train = Filter.useFilter(train, this.filter[1]);
			}
			if (resample){
//				String[] opt_filter_Resample = new String[]{"-B","1.0","-S","1","-Z","100.0"};
				//opt_filter_Resample[1] = String.valueOf(config[3]);
				this.setResampleRatio(config[3]);
				this.filter[0].setInputFormat(train);
				train = Filter.useFilter(train, this.filter[0]);
			}
			String[] str_opt_RandomForest = new String[]{"-I",Integer.toString((int)config[0]),"-K",Integer.toString((int)config[1]),"-depth",Integer.toString((int)config[2]),"-num-slots","2"};
			this.classifier.setOptions(str_opt_RandomForest);
			this.classifier.buildClassifier(train);
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(this.classifier, test);
			
			for (int classIndex=0; classIndex<=1; classIndex+=1){
				result[ii][classIndex*5+0] = eval.truePositiveRate(classIndex);
				result[ii][classIndex*5+1] = eval.falsePositiveRate(classIndex);
				result[ii][classIndex*5+2] = eval.precision(classIndex);
				result[ii][classIndex*5+3] = eval.recall(classIndex);
				result[ii][classIndex*5+4] = eval.fMeasure(classIndex);
				result[ii][classIndex*2+15] = eval.confusionMatrix()[classIndex][0];
				result[ii][classIndex*2+16] = eval.confusionMatrix()[classIndex][1];
			}
			result[ii][10] = eval.weightedTruePositiveRate();
			result[ii][11] = eval.weightedFalsePositiveRate();
			result[ii][12] = eval.weightedPrecision();
			result[ii][13] = eval.weightedRecall();
			result[ii][14] = eval.weightedFMeasure();
			fw.write(Arrays.toString(result[ii])+"\n");
			fw.close();
		}
		DescriptiveStatistics ds = new DescriptiveStatistics();
		double[] result_mean = new double[result[0].length];
		double[] result_std = new double[result[0].length];
		for (int ii=0; ii<result[0].length;ii++){
			for (int jj = 0; jj<result.length;jj++){
				ds.addValue(result[jj][ii]);
			}
			result_mean[ii] = ds.getMean();
			result_std[ii] = ds.getStandardDeviation();
			ds.clear();
		}
		FileWriter fw = new FileWriter(file_name, true);
		fw.write(Arrays.toString(result_mean)+"\n");
		fw.write(Arrays.toString(result_std)+"\n");
		fw.close();
	}
	
	public void test_RF_clustered(int folds) throws Exception{
		String file_name = "./data_source/result_test.dat";
		double[][] result = new double[folds][19];
		for (int ii = 0; ii<folds; ii++){
			FileWriter fw = new FileWriter(file_name, true);
			this.data_nominal = this.data_full.trainCV(folds, ii);
			this.data_test = this.data_full.testCV(folds,ii);
			int[] config = this.gridSearch_RF_clustered(folds-1);
			fw.write(Arrays.toString(config)+"\n");
			Instances train = this.data_nominal;
			Instances test = this.data_test;
			train.setClassIndex(train.attribute("突发投诉").index());
			test.setClassIndex(test.attribute("突发投诉").index());

			boolean resample = (config[3]%2 == 1);
			boolean reweight = (config[3]/2==1);
			
			double[] result_fold = this.one_pass_RF_cluster(train, test, reweight, resample, config[4]);
			
			for (int classIndex=0; classIndex<=1; classIndex+=1){
				result[ii][classIndex*5+0] = result_fold[classIndex*5+0];
				result[ii][classIndex*5+1] = result_fold[classIndex*5+1];
				result[ii][classIndex*5+2] = result_fold[classIndex*5+2];
				result[ii][classIndex*5+3] = result_fold[classIndex*5+3];
				result[ii][classIndex*5+4] = result_fold[classIndex*5+4];
				result[ii][classIndex*2+15] = result_fold[classIndex*2+15];
				result[ii][classIndex*2+16] = result_fold[classIndex*2+16];
			}
			result[ii][10] = result_fold[10];
			result[ii][11] = result_fold[11];
			result[ii][12] = result_fold[12];
			result[ii][13] = result_fold[13];
			result[ii][14] = result_fold[14];
			fw.write(Arrays.toString(result[ii])+"\n");
			fw.close();
		}
		DescriptiveStatistics ds = new DescriptiveStatistics();
		double[] result_mean = new double[result[0].length];
		double[] result_std = new double[result[0].length];
		for (int ii=0; ii<result[0].length;ii++){
			for (int jj = 0; jj<result.length;jj++){
				ds.addValue(result[jj][ii]);
			}
			result_mean[ii] = ds.getMean();
			result_std[ii] = ds.getStandardDeviation();
			ds.clear();
		}
		FileWriter fw = new FileWriter(file_name, true);
		fw.write(Arrays.toString(result_mean)+"\n");
		fw.write(Arrays.toString(result_std)+"\n");
		fw.close();
	}
	
	public void test_RF_FSelected(int folds) throws Exception{
		String file_name = "./data_source/result_test.dat";
		double[][] result = new double[folds][19];
		for (int ii = 0; ii<folds; ii++){
			FileWriter fw = new FileWriter(file_name, true);
			this.data_nominal = this.data_full.trainCV(folds, ii);
			this.data_test = this.data_full.testCV(folds,ii);
//			Attribute time_test = this.data_test.attribute("具体时刻");
			Attribute time_att = this.data_full.attribute("具体时刻");
			Instances data_training_original = new Instances(this.data_nominal);
			Instances data_test_original = new Instances(this.data_test);
			att_sel =  new AttributeSelection[folds];
			if (time_att != null){
				this.data_nominal.deleteAttributeAt(time_att.index());
				this.data_test.deleteAttributeAt(time_att.index());
			}
			double[] config = this.gridSearch_RF_FSelected(folds-1);  // One fold data for test, all the others for training and verification
//			double[] config = {200,23, 6, 0.1, 0.005};  // Just for test 
			fw.write(String.format("batch %d\n",ii));
			fw.write(Arrays.toString(config)+"\n");
			Instances train = this.data_nominal;
			Instances test = this.data_test;
			train.setClassIndex(train.attribute("突发投诉").index());
			test.setClassIndex(test.attribute("突发投诉").index());
			
			AttributeSelection as = new AttributeSelection();
			GainRatioAttributeEval grae = new GainRatioAttributeEval();
			Ranker ranker = new Ranker();
			if (config[4] <= 1) {ranker.setThreshold(config[4]);}
			else {ranker.setNumToSelect((int)config[4]);}
			as.setEvaluator(grae);
			as.setSearch(ranker);
			
			Normalize normalizer = new Normalize();
			normalizer.setIgnoreClass(true);
			normalizer.setInputFormat(train);
			Instances train_tmp = Filter.useFilter(train, normalizer);
			as.SelectAttributes(train_tmp);
			
			int[] att_index = as.selectedAttributes();
			System.out.println("Selected feature length is " + att_index.length);
			Remove rm = new Remove();
			rm.setAttributeIndicesArray(att_index);
			rm.setInvertSelection(true);
			rm.setInputFormat(train);
			train = Filter.useFilter(train, rm);
			System.out.println("The number of attributes for selected training set is" + train.numAttributes());
			test = Filter.useFilter(test, rm);			
			
			// The following part is for the generation of text forest, on which the threshold for some feature (call drop ratio, etc) may be too small
			// for loading into data structure in RandomForestTrack. Therefore we multiply each feature by 1000. This does not impact the classifier.
			MathExpression me = new MathExpression();
			me.setIgnoreRange(String.valueOf(train.classIndex()));
			me.setExpression("A*100");
			me.setInputFormat(train);
			train = Filter.useFilter(train, me);
			test = Filter.useFilter(test, me);
			Map <String, Integer> attNameIndex = this.featureNameIndex(test);

			//boolean resample = ((int)config[3]%2 == 1);
			//boolean reweight = ((int)config[3]/2==1);			
			boolean resample = true;
			boolean reweight = false; 
			// NOTE!!!! If the resample or reweight flag is modified, keep in mind to modify similar flags in function gridSearch_RF(), too.
			this.setResampleRatio(config[3]); // Set the resampling ratio to the selected value. 
			double[] result_fold = this.one_pass_RF(train, test, reweight, resample);
//			double[] result_fold= new double[24];  // Just for test
			
			String file_name_classifier = "./data_source/forest_example.dat";
			FileWriter fw_rf = new FileWriter(file_name_classifier,false);
			fw_rf.write(this.classifier.toString());
			fw_rf.close();			
			RandomForestTrack rft= new RandomForestTrack();
//			rft.loadtrees("./data_source/forest_example.dat");
			
			if (time_att!=null){
				String file_name_instresult = "./data_source/instance_result.dat";
				FileWriter fw_ir = new FileWriter(file_name_instresult,true);
				String file_name_instreason = "./data_source/instance_reason.dat";
				FileWriter fw_ic = new FileWriter(file_name_instreason,false);
	//			String index_of_time = test.attribute(test.numAttributes()-2).name();
	//			System.out.println(index_of_time);
	//			Attribute time_instant_att = test.attribute("投诉时点");
				for (int jj = 0; jj < test.numInstances(); jj++){
					Instance inst = test.instance(jj);
					int actual = (int)inst.classValue();
					int predict = (int) this.classifier.classifyInstance(inst);
					String time_instant = data_test_original.get(jj).stringValue(time_att.index());
					String out = String.format("actual: %d, predicted: %d, time: %s\n", actual, predict, time_instant);
					
					fw_ir.write(out);
					fw_ic.write(out);
//					String[] reasons = rft.genReason(inst, attNameIndex);
//					for (int kk = 0; kk < reasons.length; kk++){
//						if (reasons[kk]!=null){
//							fw_ic.write(String.format("Tree %d: %s\n", kk+1, reasons[kk]));
//						}
//					}
				}
			
	//			fw_ir.write(this.classifier.toString());
				fw_ir.close();
				fw_ic.close();
			}
			
			
			for (int classIndex=0; classIndex<=1; classIndex+=1){
				result[ii][classIndex*5+0] = result_fold[classIndex*5+0];
				result[ii][classIndex*5+1] = result_fold[classIndex*5+1];
				result[ii][classIndex*5+2] = result_fold[classIndex*5+2];
				result[ii][classIndex*5+3] = result_fold[classIndex*5+3];
				result[ii][classIndex*5+4] = result_fold[classIndex*5+4];
				result[ii][classIndex*2+15] = result_fold[classIndex*2+15];
				result[ii][classIndex*2+16] = result_fold[classIndex*2+16];
			}
			result[ii][10] = result_fold[10];
			result[ii][11] = result_fold[11];
			result[ii][12] = result_fold[12];
			result[ii][13] = result_fold[13];
			result[ii][14] = result_fold[14];
			fw.write(Arrays.toString(result[ii])+"\n");
			fw.close();
		}
		DescriptiveStatistics ds = new DescriptiveStatistics();
		double[] result_mean = new double[result[0].length];
		double[] result_std = new double[result[0].length];
		for (int ii=0; ii<result[0].length;ii++){
			for (int jj = 0; jj<result.length;jj++){
				ds.addValue(result[jj][ii]);
			}
			result_mean[ii] = ds.getMean();
			result_std[ii] = ds.getStandardDeviation();
			ds.clear();
		}
		FileWriter fw = new FileWriter(file_name, true);
		fw.write(Arrays.toString(result_mean)+"\n");
		fw.write(Arrays.toString(result_std)+"\n");
		fw.close();
	}
	
	public Map <String, Integer> featureNameIndex(Instances  for_prediction){
		Map <String, Integer>  ml = new HashMap<String, Integer> ();
		for (int ii = 0; ii < for_prediction.numAttributes(); ii++){
			Attribute att = for_prediction.attribute(ii);
			String att_name = att.name();
			ml.put(att_name, ii);
		}
		return ml;
	}
	
	public Instances loadTestSet(String data_file_name) throws Exception{
//		String data_file_name = "./data_source/training_set_达州_2G互联网_有信号但无法使用_thre=5_randomized.csv";
		//String data_file_name = "./data_source/test.csv";
//		String data_file_name = "./data_source/training_set_CS_coverage_thre=8_randomized.csv";
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(data_file_name));
		Instances data_raw = loader.getDataSet();
		String[] opt_filter_nominal = new String[2];
		opt_filter_nominal[0] = "-R";
//		opt_filter_nominal[1] = "last";
		opt_filter_nominal[1] = String.valueOf(data_raw.attribute("突发投诉").index()+1); // Note: Here the index shall be the real one plus 1, since in the input format, index starts from 1 instead of 0
		NumericToNominal num2nom = new NumericToNominal();
		num2nom.setOptions(opt_filter_nominal);
		num2nom.setInputFormat(data_raw);
		Instances data_nominal = Filter.useFilter(data_raw, num2nom);
		return data_nominal;
	}
	
	public void overallModelTesting(int folds, String testsetname) throws Exception{
		String file_name = "./data_source/result_test_July.dat";
		double[] result = new double[19];
		FileWriter fw = new FileWriter(file_name, true);
		this.data_nominal = this.data_full;
		Instances data_test = this.loadTestSet(testsetname);
		Attribute time_att = this.data_full.attribute("具体时刻");
		Instances data_test_original = new Instances(data_test);
		if (time_att != null){
			this.data_nominal.deleteAttributeAt(time_att.index());
			data_test.deleteAttributeAt(time_att.index());
		}
		double[] config = this.gridSearch_RF_FSelected(folds);  // One fold data for test, all the others for training and verification
//			double[] config = {200,23, 6, 0.1, 0.005};  // Just for test 
		fw.write(Arrays.toString(config)+"\n");
		Instances train = this.data_nominal;
		Instances test = data_test;
		train.setClassIndex(train.attribute("突发投诉").index());
		test.setClassIndex(test.attribute("突发投诉").index());
		
		AttributeSelection as = new AttributeSelection();
		GainRatioAttributeEval grae = new GainRatioAttributeEval();
		Ranker ranker = new Ranker();
		if (config[4] <= 1) {ranker.setThreshold(config[4]);}
		else {ranker.setNumToSelect((int)config[4]);}
		as.setEvaluator(grae);
		as.setSearch(ranker);
		
		Normalize normalizer = new Normalize();
		normalizer.setIgnoreClass(true);
		normalizer.setInputFormat(train);
		Instances train_tmp = Filter.useFilter(train, normalizer);
		as.SelectAttributes(train_tmp);
		
		int[] att_index = as.selectedAttributes();
		System.out.println("Selected feature length is " + att_index.length);
		Remove rm = new Remove();
		rm.setAttributeIndicesArray(att_index);
		rm.setInvertSelection(true);
		rm.setInputFormat(train);
		train = Filter.useFilter(train, rm);
		System.out.println("The number of attributes for selected training set is" + train.numAttributes());
		test = Filter.useFilter(test, rm);			
		
		// The following part is for the generation of text forest, on which the threshold for some feature (call drop ratio, etc) may be too small
		// for loading into data structure in RandomForestTrack. Therefore we multiply each feature by 1000. This does not impact the classifier.
		MathExpression me = new MathExpression();
		me.setIgnoreRange(String.valueOf(train.classIndex()));
		me.setExpression("A*100");
		me.setInputFormat(train);
		train = Filter.useFilter(train, me);
		test = Filter.useFilter(test, me);
		Map <String, Integer> attNameIndex = this.featureNameIndex(test);

		//boolean resample = ((int)config[3]%2 == 1);
		//boolean reweight = ((int)config[3]/2==1);			
		boolean resample = true;
		boolean reweight = false; 
		// NOTE!!!! If the resample or reweight flag is modified, keep in mind to modify similar flags in function gridSearch_RF(), too.
		this.setResampleRatio(config[3]); // Set the resampling ratio to the selected value. 
		double[] result_fold = this.one_pass_RF(train, test, reweight, resample);
//			double[] result_fold= new double[24];  // Just for test
		
		String file_name_classifier = "./data_source/forest_example.dat";
		FileWriter fw_rf = new FileWriter(file_name_classifier,false);
		fw_rf.write(this.classifier.toString());
		fw_rf.close();			
		RandomForestTrack rft= new RandomForestTrack();
//		rft.loadtrees("./data_source/forest_example.dat");
		
		if (time_att!=null){
			String file_name_instresult = "./data_source/instance_result.dat";
			FileWriter fw_ir = new FileWriter(file_name_instresult,false);
			String file_name_instreason = "./data_source/instance_reason.dat";
			FileWriter fw_ic = new FileWriter(file_name_instreason,false);
//			String index_of_time = test.attribute(test.numAttributes()-2).name();
//			System.out.println(index_of_time);
//			Attribute time_instant_att = test.attribute("投诉时点");
			for (int jj = 0; jj < test.numInstances(); jj++){
				Instance inst = test.instance(jj);
				int actual = (int)inst.classValue();
				int predict = (int) this.classifier.classifyInstance(inst);
				String time_instant = data_test_original.get(jj).stringValue(time_att.index());
				String out = String.format("actual: %d, predicted: %d, time: %s\n", actual, predict, time_instant);
				
				fw_ir.write(out);
				fw_ic.write(out);
//				String[] reasons = rft.genReason(inst, attNameIndex);
//				for (int kk = 0; kk < reasons.length; kk++){
//					if (reasons[kk]!=null){
//						fw_ic.write(String.format("Tree %d: %s\n", kk+1, reasons[kk]));
//					}
//				}
			}
		
//			fw_ir.write(this.classifier.toString());
			fw_ir.close();
			fw_ic.close();
		}		
		fw.write(Arrays.toString(result_fold)+"\n");
		fw.close();
	}
	
	public static void main(String[] args) throws Exception {
		MacroBurstPred mbp = new MacroBurstPred();
		mbp.dataLoad();
		mbp.test_RF_FSelected(6);
//		mbp.overallModelTesting(5, "./data_source/testing_set_成都_2G互联网_有信号但无法使用_thre=13_randomized_periodic_1hour.csv");
		
//		mbp.train_RF_clustered(5, 5, 23, 250, true, true);
//		mbp.gridSearch();
//		System.out.println(Arrays.toString(result));
	}

}
