package wekaTest;

import java.util.Stack;
import java.util.*;
import weka.core.Instance;

public class TextTree {
    TreeNode root = null;
	private double size = 0;
	
	public TextTree(){
		
	}
	
	public boolean insert (String feature_self, double thre_self, int judge_self, String feature_parent, double thre_parent)
	// This method defines how a new node is included into the decision tree. If judge_self == -1, it means the new node is 
	// not a leaf node, else it's a leaf node. The method handles the 2 cases in different ways
	{
		if (root == null){
			root = new TreeNode(feature_self, thre_self,  judge_self);
			size++;
			if (judge_self!=-1){
				TreeNode leaf_result = new TreeNode("NULL", 0, judge_self);
				if(root.leftChild==null)
					root.leftChild = leaf_result;
				else
					root.rightChild = leaf_result;
				leaf_result.parentNode = root;
			}
		}
		else{
			if (judge_self==-1){  // No decision result, not a leaf node
				TreeNode node_exist = new TreeNode();
//				TreeNode node = this.noRecPostOrder(feature_parent, thre_parent);
				ArrayList <TreeNode> nodes_exist = this.noRecPostOrder(feature_self, thre_self);  // Check if the node already exists. May return many ones who share the same feature name and threshold
				ArrayList <TreeNode> nodes_parent = this.noRecPostOrder(feature_parent, thre_parent);  // Check if the parent node already exists. May return many ones who share the same feature name and threshold
				if (nodes_exist == null){   // If no nodes is returned, of course it does not exist. Create a new one.				
					node_exist = new TreeNode(feature_self, thre_self, judge_self);
					TreeNode node = new TreeNode();
					for (int kk = 0; kk < nodes_parent.size(); kk++){ // Here we find the actual parent node, since many nodes may be returned.
						node = nodes_parent.get(kk);
						if (node.leftChild == null || node.rightChild == null)  // The parent node shall have one child empty
							break;
					}
					if (node.leftChild==null)
						node.leftChild = node_exist;
					else
						node.rightChild = node_exist;
					node_exist.parentNode = node;
				}
				else{ // If node is returned, we have to consider the possibility that more than one is returned. 
					// The one with both child nodes filled shall be the wrong one. If all of them does not match, we also have to create a new node.
					boolean exist = false;
					for(int kk = 0; kk<nodes_exist.size(); kk++){
						node_exist = nodes_exist.get(kk);
						if (node_exist.leftChild != null && node_exist.rightChild!= null)
							continue;
						else{
							exist = true;  // Got a node with at least child node empty. Shall be the one we are looking for. And do nothing! :)
							break;}
					}
					if (!exist){ // If the flag is still false, it means the returned list does not include the one we are looking for. create a new one
						node_exist = new TreeNode(feature_self, thre_self, judge_self);
						TreeNode node = new TreeNode();
						for (int kk = 0; kk < nodes_parent.size(); kk++){
							node = nodes_parent.get(kk);
							if (node.leftChild == null || node.rightChild == null)
								break;
						}
						if (node.leftChild==null)
							node.leftChild = node_exist;
						else
							node.rightChild = node_exist;
						node_exist.parentNode = node;
					}
				}
			}
			else{ // Decision result provided. Shall attach leaf node to the parent decision node
				TreeNode node_exist = new TreeNode();
//				TreeNode node = this.noRecPostOrder(feature_parent, thre_parent);
				ArrayList <TreeNode> nodes_exist = this.noRecPostOrder(feature_self, thre_self);  // Check if the node already exists. May return many ones who share the same feature name and threshold
				ArrayList <TreeNode> nodes_parent = this.noRecPostOrder(feature_parent, thre_parent);  // Check if the parent node already exists. May return many ones who share the same feature name and threshold
				if (nodes_exist == null){   // If no nodes is returned, of course it does not exist. Create a new one.				
					node_exist = new TreeNode(feature_self, thre_self, judge_self);
					TreeNode node = new TreeNode();
					for (int kk = 0; kk < nodes_parent.size(); kk++){ // Here we find the actual parent node, since many nodes may be returned.
						node = nodes_parent.get(kk);
						if (node.leftChild == null || node.rightChild == null)  // The parent node shall have one child empty
							break;
					}
					if (node.leftChild==null)
						node.leftChild = node_exist;
					else
						node.rightChild = node_exist;
					node_exist.parentNode = node;
				}
				else{ // If node is returned, we have to consider the possibility that more than one is returned. 
					// The one with both child nodes filled shall be the wrong one. If all of them does not match, we also have to create a new node.
					boolean exist = false;
					for(int kk = 0; kk<nodes_exist.size(); kk++){
						node_exist = nodes_exist.get(kk);
						if (node_exist.leftChild != null && node_exist.rightChild!= null)
							continue;
						else{
							exist = true;  // Got a node with at least child node empty. Shall be the one we are looking for. And do nothing! :)
							break;}
					}
					if (!exist){ // If the flag is still false, it means the returned list does not include the one we are looking for. create a new one
						node_exist = new TreeNode(feature_self, thre_self, judge_self);
						TreeNode node = new TreeNode();
						for (int kk = 0; kk < nodes_parent.size(); kk++){
							node = nodes_parent.get(kk);
							if (node.leftChild == null || node.rightChild == null)
								break;
						}
						if (node.leftChild==null)
							node.leftChild = node_exist;
						else
							node.rightChild = node_exist;
						node_exist.parentNode = node;
					}
				}
 				TreeNode leaf_result = new TreeNode("NULL", 0, judge_self);
				if(node_exist.leftChild==null)
					node_exist.leftChild = leaf_result;
				else
					node_exist.rightChild = leaf_result;
				leaf_result.parentNode = node_exist;
			}
			size++;
		}
		return true;
	}
	
	public ArrayList <TreeNode> noRecPostOrder(String a, double b){
		Stack<TreeNode> stack = new Stack<TreeNode>();
		ArrayList <TreeNode> nodes_found = new ArrayList();
		TreeNode p = root;
		TreeNode node = p;
		while (p!=null){
			for(;p.leftChild!=null;p=p.leftChild){
				stack.push(p);
			}
			while(p!=null&&(p.rightChild==null||p.rightChild==node)){
				node = p;
//				if(stack.empty())
//					return node;
				if (p.feature_name.equals(a) && p.threshold == b)
					nodes_found.add(p);
				if (stack.empty())
					if (nodes_found.isEmpty())
						return null;
					else
						return nodes_found;
				p = stack.pop();
			}
			stack.push(p);
			p= p.rightChild;
		}
//		System.out.println("The node matching the input feature and threshold does no exist");
		return nodes_found;
	}
	
	public String getReason(Instance instance, Map map){
		TreeNode p = root;
		String output = "";
//		System.out.println(root.feature_name);
		try{
			while(p.feature_name!="NULL"){
				if (root.feature_name == "RRU组网级数与配置不一致告警_1")
					System.out.println(p.feature_name);
				double inst_value = instance.value((int) map.get(p.feature_name));
				if(inst_value < p.threshold){
					output = output.concat(p.feature_name.concat(String.format(" %1.3f < %1.3f,", inst_value/100, p.threshold/100)));
					p = p.leftChild;
				}
				else{
					output = output.concat(p.feature_name.concat(String.format(" %1.3f >= %1.3f,", inst_value/100, p.threshold/100)));
					p = p.rightChild;
				}
			}
		} catch (NullPointerException e){
			System.out.println(root.feature_name);
			System.out.println(output);
			System.out.println(p!=null);
		}
		output = output.concat(String.format("%d", p.type));
		return output;
	}
	
	static class TreeNode{
		protected String feature_name;
		protected double threshold;
		protected int type;
		protected TreeNode parentNode;
		protected TreeNode leftChild;
		protected TreeNode rightChild;
		
		public TreeNode(){}
		
		public TreeNode(String feature_name, double threshold, int type){
			this.feature_name = feature_name;
			this.threshold = threshold;
			this.type = type;
		}
	}
}
