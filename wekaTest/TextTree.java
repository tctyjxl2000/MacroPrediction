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
				TreeNode node_exist = this.noRecPostOrder(feature_self, thre_self);  // Check if the node already exists
				if (node_exist == null){   // If the node does not exist, create a new one. Else do nothing
					TreeNode node = this.noRecPostOrder(feature_parent, thre_parent);
					TreeNode tmp_node = new TreeNode(feature_self, thre_self, judge_self);
					if (node.leftChild==null)
						node.leftChild = tmp_node;
					else
						node.rightChild = tmp_node;
					tmp_node.parentNode = node;
				}
			}
			else{ // Decision result provided. Shall attach leaf node to the parent decision node
				TreeNode node_exist = this.noRecPostOrder(feature_self, thre_self);
				if (node_exist == null){   // If the node does not exist, create a new one.
					TreeNode node = this.noRecPostOrder(feature_parent, thre_parent);
					node_exist = new TreeNode(feature_self, thre_self, -1);
					if (node.leftChild==null)
						node.leftChild = node_exist;
					else
						node.rightChild = node_exist;
					node_exist.parentNode = node;
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
	
	public TreeNode noRecPostOrder(String a, double b){
		Stack<TreeNode> stack = new Stack<TreeNode>();
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
					return p;
				if (stack.empty())
					return null;
				p = stack.pop();
			}
			stack.push(p);
			p= p.rightChild;
		}
		System.out.println("The node matching the input feature and threshold does no exist");
		return null;
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
					output = output.concat(p.feature_name.concat(String.format("%1.3f < %1.3f,",inst_value/100000, p.threshold/100000)));
					p = p.leftChild;
				}
				else{
					output = output.concat(p.feature_name.concat(String.format("%1.3f >= %1.3f,",inst_value/100000, p.threshold/100000)));
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
