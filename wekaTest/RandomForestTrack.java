package wekaTest;

import java.io.*;
import java.util.regex.*;
import java.util.*;
import wekaTest.TextTree;
import weka.core.*;

public class RandomForestTrack {
	TextTree[] Trees = new TextTree[500];
	
	public void loadtrees(String printedtrees_file) throws Exception{
		File filename= new File(printedtrees_file);
		InputStreamReader reader = new InputStreamReader(new FileInputStream(filename));
		BufferedReader br = new BufferedReader(reader);
//		TextTree[] Trees = new TextTree[500];
		String[] feature_name = new String[20];
		double[] thre = new double[20];
		int tree_index = -1;
		
		String line = "";
		line = br.readLine();
		while (line != null){
			if (line.contains("RandomTree")){
				feature_name = new String[20];
				thre = new double[20];
				tree_index++;
				if (tree_index == 15)
					System.out.println(tree_index);
				TextTree tmpTree = new TextTree();
				int class_type = -1;
				
				line = br.readLine();			// The line"========"
				line = br.readLine();         // The empty line
				line = br.readLine();        // This is what we want
				while(line != null && !(line.contains("Size of the tree"))){
					Pattern p = Pattern.compile("\\|   ");
					String[] tmp= p.split(line);
					int level = tmp.length-1;
					String aaa = tmp[tmp.length-1];
					String[] decision_elements = Pattern.compile("\\ < |\\ >= |\\ : ").split(aaa);
					switch (decision_elements.length){
						case 1: level= -1; break;
						case 2: {
							feature_name[level] = decision_elements[0]; 
							try{
								thre[level] = Double.valueOf(decision_elements[1]);
							}catch(NumberFormatException e){
								System.out.println("The input threshold value is not legal");
							}							
							if (feature_name[level].equals("EDGE性能数据_RLC层下行GPRS流量_2") && thre[level]==945.36)
								System.out.println(tree_index);
							class_type = -1;
							break;
						}
						case 3:{
							feature_name[level] = decision_elements[0]; 
							try{
								thre[level] = Double.valueOf(decision_elements[1]);
							}catch(NumberFormatException e){
								System.out.println("The input threshold value is illegal");
							}
							try{
								String[] decision_result = Pattern.compile(" \\(").split(decision_elements[2]);
								class_type = Integer.valueOf(decision_result[0]);
							}catch(NumberFormatException e){
								System.out.println("The decision result value is illegal");
							}
							break;
						}
					}
					if (level>0)
						tmpTree.insert(feature_name[level], thre[level], class_type, feature_name[level-1], thre[level-1]);
					else if (level==0)
						tmpTree.insert(feature_name[level], thre[level], class_type, "not existing", 0);
					line = br.readLine();
				}
				Trees[tree_index] = tmpTree;
			}			
			line = br.readLine();
		}
	}
	
	public String[] genReason(Instance instance, Map<String, Integer> ml){
		String[] reasons = new String[Trees.length];
		for(int ii = 0; ii<Trees.length; ii++){
			TextTree tree = Trees[ii];
			if (tree!=null)
				try{
				reasons[ii] = tree.getReason(instance, ml);
				}catch (NullPointerException e){
					System.out.println(String.format("Problem met at the %d-th tree", ii));
				}
		}
		Map <String, Integer> cause_set_p = new HashMap <String, Integer> ();
		Map <String, ArrayList> cause_range_p = new HashMap <String, ArrayList> ();
		Map <String, Integer> cause_set_n = new HashMap <String, Integer> ();
		Map <String, ArrayList> cause_range_n = new HashMap <String, ArrayList> ();
		for (int ii = 0;ii < reasons.length; ii++){
			if (reasons[ii]!=null){
				String[] causes = reasons[ii].split(",");
				if (causes[causes.length-1].equals("1")){  // If this is a positive instance
					for (int jj = 0; jj < causes.length-1; jj++){
						String[] feature = Pattern.compile("\\ < |\\ >=").split(causes[jj]);  // feature[0], name of the feature; feature[1], threshold of the feature
						if (feature.length<2)
							System.out.println(causes[jj]);
						if (cause_set_p.containsKey(feature[0])){
							cause_set_p.replace(feature[0], cause_set_p.get(feature[0])+1);
							ArrayList <String> tmp_list = cause_range_p.get(feature[0]);
							tmp_list.add(feature[1]);
							cause_range_p.replace(feature[0], tmp_list);
						}
						else{   
							cause_set_p.put(feature[0], 1);
							ArrayList <String> tmp_list = new ArrayList <String> ();
							tmp_list.add(feature[1]);
							cause_range_p.put(feature[0], tmp_list);
						}
					}
				}
				else{  // If this is a negative instance
					for (int jj = 0; jj < causes.length-1; jj++){
						String[] feature = Pattern.compile("\\ < |\\ >=").split(causes[jj]);
						if (cause_set_n.containsKey(feature[0])){
							cause_set_n.replace(feature[0], cause_set_n.get(feature[0])+1);
							ArrayList <String> tmp_list = cause_range_n.get(feature[0]);
							tmp_list.add(feature[1]);
							cause_range_n.replace(feature[0], tmp_list);
						}
						else{
							cause_set_n.put(feature[0], 1);
							ArrayList <String> tmp_list = new ArrayList <String> ();
							tmp_list.add(feature[1]);
							cause_range_n.put(feature[0], tmp_list);
						}
					}
				}
			}
		else
			break;
		}
		return reasons;
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		RandomForestTrack rft= new RandomForestTrack();
		rft.loadtrees("./data_source/forest_example.dat");

	}

}
