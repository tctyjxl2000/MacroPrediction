package wekaTest;

import java.io.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.core.Instances;
import weka.core.converters.DatabaseLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveFolds;


public class RealTimeData extends Thread {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public Instances loadData(String args) throws Exception {
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
		return testing;
	}
	
	public void outtoFile(String args, Instances instanceInput) throws Exception {
		
		try {
			FileWriter file = new FileWriter(args);
			BufferedWriter bw = new BufferedWriter(file);
			for(int ii=0;ii<instanceInput.numInstances();ii++){
				bw.write(instanceInput.instance(ii).toString());
				bw.newLine();
				bw.flush();
				Thread.sleep(1000);
			}
			bw.close();
		} 
		catch (IOException e){
			System.out.println("Reading File error!");
		}
	}
	public void run(){
		try{
			Instances aa = this.loadData("aa");
			outtoFile("/home/zm/workspace/test/base_components/realtimeinput.dat",aa);
		}
		catch (Exception e){
			System.out.println("loading data error");
		}
	}
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		/*RealTimeData template = new RealTimeData();
		Instances aa = template.loadData("aa");
		template.outtoFile("/home/zm/workspace/test/base_components/realtimeinput.dat",aa);*/
		RealTimeData template = new RealTimeData();
		template.start();
	}

}
