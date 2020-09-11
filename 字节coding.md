## 字节coding

#### 买卖股票的最佳时机

```java
dp[i][k][0 or 1]
i代表天数
k代表允许交易的最大次数
0 or 1代表当前的持有状态(0代表没有，1代表持有)

base case：
dp[-1][k][0] = dp[i][0][0] = 0
dp[-1][k][1] = dp[i][0][1] = -infinity

状态转移方程：
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])

    
//1次交易  
class Solution{
  public int maxProfit(int[] prices){
    if(prices==null||prices.length==0){
      return 0;
    }
	int len = prices.length;
	int[][] dp=new int[len+1][2];
	dp[0][0]=0;
	dp[0][1]=Integer.MIN_VALUE;
    for(int i=1;i<=len;i++){
        dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i-1]); 
    	dp[i][1]=Math.max(dp[i-1][1],-prices[i-1]);
    }
    return dp[len][0];
  }
}

//不限制次数
dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i-1])
dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i-1])
  
//有一天冷却期
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i-1])
dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i-1])
  
//2次买卖
dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])

int n=prices.length;
int max_k = 2;
int[][][] dp = new int[n][max_k + 1][2];
for (int i = 0; i < n; i++) {
  for (int k = max_k; k >= 1; k--) {
    if (i - 1 == -1) { 
      /* 处理 base case */
      dp[i][k][0] = 0;
      dp[i][k][1] = -prices[i];
      continue;
    }
    dp[i][k][0] = Math.max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
    dp[i][k][1] = Math.max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
  }
}
// 穷举了 n × max_k × 2 个状态，正确。
return dp[n - 1][max_k][0];
```



#### 剪绳子

给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

```java
class Solution{
  public int cutRope(int target){
    int[] dp = new int[target+1];
    dp[1]=1;
    dp[2]=1;
    for(int i=3;i<=target;i++){
      for(int k=2;k<i;k++){
        dp[i]=Math.max(dp[i],Math.max(k*(i-k),dp[i-k]*k));
      }
    }
    return dp[target];
  }
}

//考虑越界
class Solution {
    public int cuttingRope(int n) {
        if(n <= 3) return n - 1;
        long res=1L;
        int p=(int)1e9+7;
        //贪心算法，优先切三，其次切二
        while(n>4){
            res=res*3%p;
            n-=3;
        }
        //出来循环只有三种情况，分别是n=2、3、4
        return (int)(res*n%p);
    }
}
```



#### 接雨水

```java
class Solution{
  public int trap(int[] height){
    if(height==null||height.length<3){
      return 0;
    }
    //接雨水的数量
    int sum = 0;
    //left[i]表示第i列左边的最高的列值
    int[] left=new int[height.length];
    left[0]=height[0];
    for(int i=1;i<height.length;i++){
      left[i]=left[i-1]>height[i]?left[i-1]:height[i];
    }
    //right[i]表示第i列右边的最高的列值
    int[] right=new int[height.length];
    right[height.length-1]=height[right.length-1];
    for(int i=height.length-2;i>=0;i--){
      right[i]=right[i+1]>height[i]?right[i+1]:height[i];
    }
    for(int i=1;i<height.length-1;i++){
      sum+=Math.min(left[i],right[i])-height[i];
    }
    return sum;
  }
}
```



#### 柠檬水找零

```java
class Solution{
  public boolean lemonadeChange(int[] bills){
    int five=0,ten=0;
    for(int value:bills){
      if(value==5){
        five++;
      }else if(value==10){
        if(five==0) return false;
        five--;
        ten++;
      }else{
        if(ten>=1 && five >=1){
          ten--;
          five--;
        }else if(five>=3){
          five=five-3;
        }else{
          return false;
        }
      }
    }
    return true;
  }
}
```



#### 双指针遍历

解决有序数组的问题

排序数组，平方后，数组当中有多少不同的数字（相同算一个）

一个数据先递增再递减，找出数组不重复的个数，比如 [1, 3, 9, 1]，结果为3，不能使用额外空间，复杂度o(n)

```java
class Solution{
  public int diffSquareNum(int[] nums){
    if(nums==null||nums.length==0){
      return 0;
    }
    int n = nums.length;
    int sum=0;
    int left=0;
    int right=n-1;
    while(left<=right){
      if(nums[left]+nums[right]==0){
        sum++;
        int temp=nums[left];
        while(left<=right && nums[left]==temp)
          left++;
        while(left<=right && nums[right]==-temp)
          right--;
      }else if(nums[left]+nums[right]<0){
        sum++;
        int temp=nums[left];
        while(left<=right && nums[left]==temp)
          left++;
      }else{
        sum++;
        int temp=nums[right];
        while(left<=right && nums[right]==temp)
          right--;
      }
    }
    return sum;
  }
}
```



#### 两数之和

```java
//哈希表
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> indexForNum = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (indexForNum.containsKey(target - nums[i])) {
                return new int[]{indexForNum.get(target - nums[i]), i};
            } else {
                indexForNum.put(nums[i], i);
            }
        }
        return null;
    }
}
//排序+双指针法
```



#### 滑动窗口

解决连续序列问题

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

```java
class Solution{
  public int[][] findContinuousSequence(int target){
    int i=1;
    int j=1;
    int sum=0;
    List<int[]> list=new ArrayList<>();
    while(i<=target/2){
      if(sum<target){
        sum+=j;
        j++;
      }else if(sum>target){
        sum-=i;
        i++;
      }else{
        int[] res=new int[j-i];
        for(int z=i;z<j;z++){
          res[z-i]=z;
        }
        list.add(res);
        sum-=i;
        i++;
      }
    }
    return list.toArray(new int[list.size()][]);
  }
}
```



给定m个不重复的字符 [a, b, c, d]，以及一个长度为n的字符串tbcacbdata，问能否在这个字符串中找到一个长度为m的连续子串，使得这个子串刚好由上面m个字符组成，顺序无所谓，返回任意满足条件的一个子串的起始位置，未找到返回-1。比如上面这个例子，acbd，3。

```java
class Solution{
  public int checkInclusion(char[] ch,String s){
    if(ch.length>s.length()){
      return -1;
    }
    for(int i=0;i<s.length-ch.length;i++){
      if(matches(ch,s.substring(i,i+ch.length)))
        return i;
    }
    return -1;
  }
  
  private boolean matches(char[] ch,String s){
    for(int i=0;i<s.length;i++){
      if(s.indexOf(ch[i])==-1)
        return false;
    }
    return true;
  }
}
```



求数组的最长连续递增数列，如：4， 200， 3， 1， 100， 2。结果是1 2 3 4，也就是说顺序可以打乱

```java
class Solution{
  public int longestConsecutive(int[] nums){
    if(nums==null||nums.length==0) return 0;
    HashSet<Integer> set=new HashSet<>();
    for(int value : nums)
      set.add(value);
    int longestLength = 0;
    for(int num:set){
      if(!set.contains(num-1)){
        int currentNum = num;
        int currentLength=1;
        while(set.contains(currentNum+1)){
          currentNum +=1;
          currentLength+=1;
        }
        longestLength=Math.max(longestLength,currentLength);
      }
    }
    return longestLength;
  }
}
```



实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

```java
class Solution {
    public void nextPermutation(int[] nums) {
        if(nums==null||nums.length<2){
            return;
        }
        int n=nums.length;
        for(int i=n-2;i>=0;i--){
            for(int j=n-1;j>i;j--){
                if(nums[i]<nums[j]){
                    swap(nums,i,j);
                    Arrays.sort(nums,i+1,nums.length);
                    return;
                }
            }
        }
        Arrays.sort(nums);
    }

    public void swap(int[] nums,int i,int j){
        int temp=nums[i];
        nums[i]=nums[j];
        nums[j]=temp;
    }
}
```



快速排序

```java
public static void quickSort(int[] arr,int low,int high){
  int i,j,temp,t;
  if(low>high){
    return;
  }
  i=low;
  j=high;
  //temp就是基准位
  temp = arr[low];

  while (i<j) {
    //先看右边，依次往左递减
    while (temp<=arr[j]&&i<j) {
      j--;
    }
    //再看左边，依次往右递增
    while (temp>=arr[i]&&i<j) {
      i++;
    }
    //如果满足条件则交换
    if (i<j) {
      t = arr[j];
      arr[j] = arr[i];
      arr[i] = t;
    }

  }
  //最后将基准为与i和j相等位置的数字交换
  arr[low] = arr[i];
  arr[i] = temp;
  //递归调用左半数组
  quickSort(arr, low, j-1);
  //递归调用右半数组
  quickSort(arr, j+1, high);
}
```



堆排序

```java
private int[] buildMaxHeap(int[] array) {
    //从最后一个节点array.length-1的父节点（array.length-1-1）/2开始，
    //直到根节点0，反复调整堆
    for (int i = (array.length - 2) / 2; i >= 0; i--) {
        adjustDownToUp(array, i, array.length);
    }
    return array;
}
//将元素array[k]自下往上逐步调整树形结构
private void adjustDownToUp(int[] array, int k, int length) {
    int temp = array[k];
    for (int i = 2 * k + 1; i < length - 1; i = 2 * i + 1) {    
    //i为初始化为节点k的左孩子，沿节点较大的子节点向下调整
        if (i < length && array[i] < array[i + 1]) {  
        //取节点较大的子节点的下标
            i++;   //如果节点的右孩子>左孩子，则取右孩子节点的下标
        }
        if (temp >= array[i]) {  //根节点 >=左右子女中关键字较大者，调整结束
            break;
        } else {   //根节点 <左右子女中关键字较大者
            array[k] = array[i]; //将左右子结点中较大值array[i]调整到双亲节点上
            k = i; //【关键】修改k值，以便继续向下调整
        }
    }
    array[k] = temp;  //被调整的结点的值放人最终位置
}
```



最小栈

```java
class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> min_stack;
    public MinStack() {
        stack = new Stack<>();
        min_stack = new Stack<>();
    }
    public void push(int x) {
        stack.push(x);
        if(min_stack.isEmpty() || x <= min_stack.peek())
            min_stack.push(x);
    }
    public void pop() {
        if(stack.pop().equals(min_stack.peek()))
            min_stack.pop();
    }
    public int top() {
        return stack.peek();
    }
    public int getMin() {
        return min_stack.peek();
    }
}
```



栈实现队列

```java
class MyQueue {

    private Stack<Integer> in; 
    private Stack<Integer> out; 

  	public MyQueue() {
		in= new Stack<>();
      	 out= new Stack<>();
    }
  
    public void push(int x) {
        in.push(x);
    }

    public int pop() {
      if(out.isEmpty()){
        int size=in.size();
        for(int i=0;i<size;i++){
          out.push(in.pop());
        }
      }
      return out.pop();
    }

    public int peek() {
	   if (out.isEmpty()) {
            int size = in.size();
            for (int i = 0; i < size; i++){
                out.push(in.pop());
            }
        }
        return out.peek();
    }

    public boolean empty() {
        return in.isEmpty() && out.isEmpty();
    }
}
```



队列实现一个栈

```java
class MyStack {

    LinkedList<Integer> queue;

    /** Initialize your data structure here. */
    public MyStack() {
        queue=new LinkedList<>();
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        queue.add(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
       return queue.removeLast();
    }
    
    /** Get the top element. */
    public int top() {
        int num=queue.removeLast();
        queue.add(num);
        return num;
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return queue.isEmpty();
    }
}
```



链表相交

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode pA = headA, pB = headB;
        //如果不相交，pA=pB=null
        while (pA != pB) {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;
    }
}
```



链表相交结点

```java
public class Solution {
    public ListNode EntryNodeOfLoop(ListNode pHead){
        ListNode fast = pHead;
        ListNode slow = pHead;
        if(fast == null || fast.next == null){
            return null;
        }
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow){
                break;
            }
        }
        slow = pHead;
        while(fast != slow){
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }
}
```



二叉树的后序遍历

```java
//非递归
public void postOrder(TreeNode root){
    if(root == null) return;
    Stack<TreeNode> s1 = new Stack<>();
    Stack<TreeNode> s2 = new Stack<>();
    s1.push(root);
    while(!s1.isEmpty()){
        TreeNode node = s1.pop();
        s2.push(node);
        if(node.left != null)
           s1.push(node.left);
       if(node.right != null)
           s1.push(node.right);
    }
    while(!s2.isEmpty())
        System.out.print(s2.pop().val + " ");
}
```



二叉搜索树的第K大的节点

二叉搜索树（二叉排序树）满足：根节点大于左子树，小于右子树。那么二叉树的中序遍历序列就是一个递增序列。为方便了找第K大的节点，我们可以调换左右子树的遍历顺序，这样遍历序列是一个递减序列，这样遍历到第K个元素就是我们要找的第K大的节点。

```java
class Solution {
    public int kthLargest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        int count = 0;
        while(p != null || !stack.isEmpty()){
            while(p != null){
                stack.push(p);
                p = p.right;
            }
            p = stack.pop();
            count++;
            if(count == k){
                return p.val;
            }
            p = p.left;
        }
        return 0;
    }
}
```



二叉树重建

```java
public class Solution {
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        if(pre==null||in==null) return null;
        HashMap<Integer,Integer> map=new HashMap<>();
        for(int i=0;i<in.length;i++){
            map.put(in[i],i);
        }
        return preIn(pre,0,pre.length-1,in,0,in.length-1,map);
    }
    TreeNode preIn(int[] p,int pi,int pj,int[] n,int ni,int nj,HashMap<Integer,Integer> map){
        if(pi>pj||ni>nj) return null;
        int index=map.get(p[pi]);
        TreeNode head=new TreeNode(p[pi]);
        head.left=preIn(p,pi+1,pi+index-ni,n,ni,index-1,map);
        head.right=preIn(p,pi+index-ni+1,pj,n,index+1,nj,map);
        return head;
    }
}
```



输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

```java
public class Solution {
    public ArrayList<Integer> printMatrix(int [][] matrix) {
        ArrayList<Integer> list=new ArrayList<>();
        if(matrix==null||matrix.length==0) return list;
        int m=matrix.length;
        int n=matrix[0].length;
        int top=0,bottom=m-1,left=0,right=n-1;
        while(top<=bottom&&left<=right){
            for(int i=left;i<=right;i++) list.add(matrix[top][i]);
            for(int i=top+1;i<=bottom;i++) list.add(matrix[i][right]);
            for(int i =right-1;i>=left&&top<bottom;--i) list.add(matrix[bottom][i]);
            for(int i = bottom-1;i>top&&right>left;--i) list.add(matrix[i][left]);
            ++top;++left;--bottom;--right;
        }
        return list;
    }
}
```



给定一个字符串，逐个翻转字符串中的每个单词。

```java
class Solution {
    public String reverseWords(String s) {
        String emptyStr = " ";
        String[] splits = s.trim().split(emptyStr);
        StringBuilder sb=new StringBuilder();
        //为了后面处理方法统一，先拼接上最后一个单词
        sb.append(splits[splits.length-1]);
        for(int i=splits.length-2;i>=0;i--){
            if (!splits[i].isEmpty()) {
                sb.append(emptyStr);
                sb.append(splits[i]);
            }
        }
        return sb.toString();
    }
}
```



最长不含重复字符的子字符串

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        Map<Character, Integer> map = new HashMap<>();
        int left = 0;
        int right = 0;
        int maxLength = 0;
        while (right < s.length()) {
            char ch = s.charAt(right);
            map.put(ch,map.getOrDefault(ch,0)+1);
            while(map.get(ch)>1){
                char ch1=s.charAt(left);
                map.put(ch1,map.get(ch1)-1);
                left++;
            }
            right++;
            maxLength=Math.max(maxLength,right-left);
        }
        return maxLength;
    }
}
```



加油站

在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        //将问题转化为找最大子串的起始位置。
        int result = 0;
        int sum = 0;
        int hasResult = 0;//用于判断是否有跑完全程所需的油
        for (int i = 0; i < gas.length; i++) {
            hasResult +=gas[i]-cost[i];
            if(sum > 0) {
                sum += gas[i]-cost[i];
            } else {
                sum = gas[i]-cost[i];
                result = i;
            }
        }

        return hasResult>=0? result:-1;
    }
}
```



合并区间

```java
class Solution{
  public int[][] merge(int[][] intervals){
    Arrays.sort(intervals,(v1,v2)->v1[0]-v2[0]);
    int[][] res=new int[intervals.length][2];
    int idx=-1;
    for(int[] interval:intervals){
      if(idx==-1||interval[0]>res[idx][1]){
        res[++idx]=interval;
      }else{
        res[idx][1]=Math.max(res[idx][1],interval[1]);
      }    
    }
    return Arrays.copyOf(res,idx+1);
  }
}
```



#### leetcode 102 二叉树的层序遍历

给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。 （即逐层地，从左到右访问所有节点）。

```java
class Solution{
  public List<List<Integer>> levelOrder(TreeNode root){
    List<List<Integer>> res = new ArrayList<>();
    if(root == null){
      return res;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while(!queue.isEmpty()){
      List<Integer> temp = new ArrayList<>();
      int len = queue.size();
      for(int i =0;i < len;i++){
        TreeNode node = queue.poll();
        temp.add(node.val);
        if(node.left != null){
          queue.add(node.left);
        }
        if(node.right != null){
          queue.add(node.right);
        }
      }
      res.add(temp);
    }
    return res;
  }
}
```



#### leetcode 146 LRU缓存机制

```java
class LRUCache{
  HashMap<Integer,Integer> map;
  LinkedList<Integer> list;
  int capacity;
  
  public LRUCache(int capacity){
    map = new HashMap<>();
    list = new LinkedList<>();
    this.capacity = capacity;
  }
  
  public int get(int key){
    if(map.containsKey(key)){
      	list.remove(Integer(key);
      	list.addLast(key);
      	return map.get(key);
    }
    return -1;
  }
  
  public void put(int key,int value){
    if(map.containsKey(key)){
      	list.remove(Integer(key);
      	list.addLast(key);
      	map.put(key,value);
        return;
    }
    if(list.size() == capacity){
	    map.remove(list.removeFirst());
      	map.put(key,value);
      	list.addLast(key);
    }else{
      map.put(key,value);
      list.addLast(key);
    }
  }
}
                    
//LinkedHashMap  O(1)时间复杂度
class LRUCache {

        private LinkedHashMap<Integer, Integer> hashMap;
        private int capacity;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            // accessOrder: 默认为false表示按插入顺序排序， true表示按访问顺序排序
            hashMap = new LinkedHashMap<Integer, Integer>(capacity, 0.75F,true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
                    return size() > capacity;
                }
            };
        }

        public int get(int key) {
            return hashMap.getOrDefault(key, -1);
        }

        public void put(int key, int value) {
            hashMap.put(key, value);
        }
}

```



#### leetcode 415 字符串相加

```java
class Solution {
    public String addStrings(String num1, String num2) {
        StringBuilder str = new StringBuilder();
        int carry = 0, i = num1.length() - 1, j = num2.length() - 1;
      	int sum = 0;
        while (carry == 1 || i >= 0 || j >= 0) {
            int x = i < 0 ? 0 : num1.charAt(i--) - '0';
            int y = j < 0 ? 0 : num2.charAt(j--) - '0';
          	sum = x + y + carry;
            str.append(sum % 10);
            carry = sum / 10;
        }
        return str.reverse().toString();
    }
}
```



#### leetcode 199 二叉树的右视图

```java
class Solution{
  public List<Integer> rightSideView(TreeNode root){
    List<Integer> res = new ArrayList<>();
    if(root == null){
      return res;
    }
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while(!queue.isEmpty()){
      int len = queue.size();
      TreeNode node = null;
      for(int i = 0;i < len;i++){
        node = queue.poll();
        if(node.left != null){
          queue.add(node.left);
        }
        if(node.right != null){
          queue.add(node.right);
        }
      }
      res.add(node.val);
    }
    return res;
  }
}
```



#### leetcode 113 路径总和II

```java
class Solution {
  List<List<Integer>> res = new ArrayList<>();
  public List<List<Integer>> pathSum(TreeNode root,int sum){
    List<Integer> temp = new ArrayList<>();
    if(root == null){
      return res;
    }
    helper(res,temp,root,sum);
    return res;
  }
  
  public void helper(List<List<Integer>> res,List<Integer> temp,TreeNode root,int sum){
    if(root == null){
        return;
    }
    if(root.left == null && root.right == null && root.val == sum){ 
      temp.add(root.val);
      res.add(new ArrayList<>(temp));
      temp.remove(temp.size()-1);
    }
    temp.add(root.val);
    helper(res,temp,root.left,sum-root.val);
    helper(res,temp,root.right,sum-root.val);
    temp.remove(temp.size()-1);
  }
}
```



#### leetcode 3 无重复字符的最长子串

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int i = 0;
        int j = 0;
        int len = s.length();
        HashMap<Character, Integer> map = new HashMap<>();
        int maxLength = 0;
        while (j < len) {
            char c1 = s.charAt(j);
            map.put(c1,map.getOrDefault(c1,0)+1);
            while (map.get(c1)>1) {
                char c2 = s.charAt(i);
                map.put(c2, map.get(c2) - 1);
                i++;
            }
            j++;
            maxLength =Math.max(maxLength,j-i);
        }
        return maxLength;
    }
}
```



#### leetcode 25  K个一组翻转链表

```java
class Solution{
  public ListNode reverseKGroup(ListNode head,int k){
    if(head == null || head.next==null){
      return head;
    }
    ListNode tail = head;
    for(int i = 0;i < k;i++){
      if(tail == null){
        return head;
      }
      tail = tail.next;
    }
     ListNode newHead = reverse(head,tail);
     head.next=reverseKGroup(tail,k);
	return newHead;
  }
  public ListNode reverse(ListNode head,ListNode tail){
    ListNode pre = null;
    ListNode temp = null;
    while(head != tail){
      temp = head.next;
      head.next=pre;
      pre = head;
      head = temp;
    }
    return pre;
  }
}

//非递归
public ListNode reverseKGroup(ListNode head, int k) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;

    ListNode pre = dummy;
    ListNode end = dummy;

    while (end.next != null) {
        for (int i = 0; i < k && end != null; i++) end = end.next;
        if (end == null) break;
        ListNode start = pre.next;
        ListNode next = end.next;
        end.next = null;
        pre.next = reverse(start);
        start.next = next;
        pre = start;

        end = pre;
    }
    return dummy.next;
}

private ListNode reverse(ListNode head) {
    ListNode pre = null;
    ListNode curr = head;
    while (curr != null) {
        ListNode next = curr.next;
        curr.next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}
```



#### leetcode 2 两数相加

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2){
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;
    int sum = 0;
    int carry = 0;
    int x = 0;
    int y = 0;
    while(l1 != null || l2 != null){
        x = (l1 != null) ? l1.val : 0;
        y = (l2 != null) ? l2.val : 0;
        sum = carry + x + y;
        carry = sum / 10;
        curr.next = new ListNode(sum % 10);
        curr = curr.next;
        if(l1 != null){
            l1 = l1.next;
        }
        if(l2 != null){
            l2 = l2.next;
        }
    }
    if(carry > 0){
        curr.next = new ListNode(carry);
    }
    return dummy.next;
    }
}
```



#### leetcode 958 二叉树的完全性检验

```java
class Solution {
    public boolean isCompleteTree(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode prev = root;
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.remove();
            if (prev == null && node != null)
                return false;
            if (node != null) {
                queue.add(node.left);
                queue.add(node.right);
            }
            prev = node;
        }
        return true;
    }
}
```



#### 剑指offer 42 连续子数组的最大和

```java
class Solution{
  public int maxSubArray(int[] nums){
    int len=nums.length;
    if(nums==null||len==0) return 0;
    int[] dp=new int[len];
    dp[0]=nums[0];
    int result=nums[0];
    for(int i=1;i<len;i++){
      dp[i]=Math.max(nums[i],dp[i-1]+nums[i]);
      result=Math.max(result,dp[i]);
    }
    return result;
  }
}
```



#### leetcode 347 前K个高频元素

```java
public class Solution {
    public List<Integer> topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> count = new HashMap<>();
        for (int n : nums) {
            count.put(n, count.getOrDefault(n, 0) + 1);
        }
        PriorityQueue<Integer> heap = new PriorityQueue<>((n1,n2) -> count.get(n1) - count.get(n2));
        for (int n : count.keySet()) {
            heap.add(n);
            if (heap.size() > k)
                heap.poll();
        }
        List<Integer> top_k = new LinkedList<>();
        while (!heap.isEmpty())
            top_k.add(heap.poll());
        Collections.reverse(top_k);
        return top_k;
    }
}
```



#### leetcode 101 对称二叉树

```java
class Solution {
  public boolean isSymmetric(TreeNode root){
    if(root == null){
      return true;
    }
    return helper(root.left,root.right);
  }
  
  public boolean helper(TreeNode l,TreeNode r){
    if(r == null && l == null){
      return true;
    }
    if(r == null || l == null){
      return false;
    }
    if(l.val != r.val){
      return false;
    }
    return helper(l.left,r.right) && helper(l.right,r.left);
  }
}
```



#### leetcode 1147 段式回文

```java
class Solution {
    public int longestDecomposition(String text) {
        int res = 0;
        int len = text.length();
        // 开始位置
        int l = 0;
        // 结尾位置
        int r = len - 1;
        // 临时长度，注意从1开始
        int temp_len = 1;
        while (l <= r){
            // 获取开始的字符
            char c1 = text.charAt(l);
            // 获取结尾的字符
            char c2 = text.charAt(r);
            // 结尾字符与开始不相等时就往后找
            while (c1 != c2){
                r--;
                c2 = text.charAt(r);
                temp_len++;
            }
            // 当开始与结尾重合时，说明这一段不能再分，只有一段
            if (l == r){
                res++;
                // 跳出循环，不再判断
                break;
            }
            // 判断两段是否相等，相等则分段
            if (text.substring(l,l+temp_len).equals(text.substring(r,r+temp_len))){
                res += 2;
                // 移动开始的位置
                l = l + temp_len;
                // 移动结尾的位置
                r--;
                // 重置临时长度
                temp_len = 1;
            }else {
                // 两段不相等，则后边继续往前
                r--;
                temp_len++;
            }
        }
        return res;
    }
}
```



#### leetcode 20 有效的括号

```java
class Solution{
  private static final Map<Character,Character> map = new HashMap<Character,Character>(){{
    put('(',')');
    put('[',']');
    put('{','}');
    put('?','?');
  }};
  
  public boolean isValid(String s){
    if(s.length() > 0 && !map.containsKey(s.charAt(0))){
      return false;
    }
    Stack<Character> stack = new Stack<Character>{{
      push('?');
    }};
    for(Character c:s.toCharArray()){
      if(map.containsKey(c)){
        stack.push(c);
      }else if(map.get(stack.pop()) != c){
        return false;
      }
    }
    return stack.size() == 1;
  }
}
```



#### leetcode 283  移动零

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int index=0;
        for(int num:nums){
            if(num!=0){
            nums[index++]=num;
            }
        }
        while(index<nums.length){
            nums[index++]=0;
        }
    }
}
```



#### leetcode 876 链表的中间结点

```java
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}
```



#### leetcode 460 LFU缓存



#### leetcode 206 反转链表

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev=null;
        ListNode cur=head;
        while(cur!=null){
            ListNode temp=cur.next;
            cur.next=prev;
            prev=cur;
            cur=temp;
        }
         return prev;
    }
}
```



#### leetcode 100 相同的树

```java
class Solution{
  public boolean isSameTree(TreeNode p,TreeNode q){
    if(p == null && q==null){
      return true;
    }
    if(p == null || q==null){
      return false;
    }
    if(p.val != q.val){
      return false;
    }
    return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
  }
}
```



#### leetcode 69 x的平方根

```java
class Solution {
    public int mySqrt(int x) {
        int l=0,h=x,ans=-1;
        while(l<=h){
            int mid=l+(h-l)/2;
            if((long)mid*mid<=x){
                ans=mid;
                l=mid+1;
            }else{
                h=mid-1;
            }
        }
        return ans;
    }
}
```



#### leetcode 124 二叉树中的最大路径和

给定一个**非空**二叉树，返回其最大路径和。

本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径**至少包含一个**节点，且不一定经过根节点。

```java
class Solution{
  int res = Integer.MIN_VALUE;
  public int maxPathSum(TreeNode root){
    if(root == null){
      return 0;
    }
    dfs(root);
    return res;
  }
  
  public int dfs(TreeNode root){
    if(root == null){
      return 0;
    }
    int leftMax = Math.max(0,dfs(root.left));
    int rightMax = Math.max(0,dfs(root.right));
    res = Math.max(res,root.val + leftMax + rightMax);
    return root.val + Math.max(leftMax,rightMax);
  }
}
```



#### 剑指offer 11 旋转数组的最小数字

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

```java
class Solution{
  public int minArray(int[] numbers){
    int i = 0,j = numbers.length - 1;
    while(i < j){
      int m = (i + j)/2;
        if(numbers[m] > numbers[j]){
          i = m + 1;
        }else if(numbers[m] < numbers[j]){
          j = m;
        }else{
          j--;
        }
    }
    return numbers[i];
  }
}
```



#### leetcode 160 相交链表

```java
public class Solution{
  public ListNode getIntersectionNode(ListNode headA,ListNode headB){
    if(headA==null||headB==null){
      return null;
    }
    ListNode p=headA,q=headB;
    while(p != q){
    	p = p==null?headB:p.next;
      	q = q==null?headA:q.next;
    }
    return p;
  }
}
```



#### leetcode15 三数之和

```java
class Solution{
  public List<List<Integer>> threeSum(int[] nums){
    Arrays.sort(nums);
    List<List<Integer>> res = new ArrayList<>();
    for(int k = 0;k < nums.length - 2;k++){
      if(nums[k] > 0){
        break;
      }
      if(k > 0 && nums[k]==nums[k-1]){
        continue;
      }
      int i = k + 1,j = nums.length - 1;
      while(i < j){
        int sum = nums[k] + nums[i] + nums[j];
        if(sum < 0){
			i++;
        }else if(sum > 0){
			j--;
        }else{
          res.add(new ArrayList<Integer>(Arrays.asList(nums[k],nums[i],nums[j])));
          while(i < j && nums[i]==nums[i+1]){
            i++;
          }
          while(i < j && nums[j]==nums[j-1]){
            j--;
          }
          i++;
          j--;
        }
      }
    }
    return res;
  }
}
```



#### leetcode 143 重排链表

```java
class Solution{
  public void reorderList(ListNode head){
    if(head == null||head.next == null){
      return;
    }
    ListNode slow = head;
    ListNode fast = head;
    while(fast.next !=null && fast.next.next !=null){
      slow = slow.next;
      fast = fast.next.next;
    }
    ListNode pre = slow;
    ListNode cur = slow.next;
    while(cur.next != null){
      ListNode tmp = cur.next;
      cur.next=tmp.next;
      tmp.next=pre.next;
      pre.next=tmp;
    }
    ListNode p1 = head;
    ListNode p2 = slow.next;
    while(p1 != slow){
      slow.next = p2.next;
      p2.next = p1.next;
      p1.next=p2;
      p1=p2.next;
      p2=slow.next;
    }
  }
}
```



#### leetcode 8  字符串转换整数

```java
public class Solution {
    public int myAtoi(String str) {
        char[] chars = str.toCharArray();
        int n = chars.length;
        int idx = 0;
        while (idx < n && chars[idx] == ' ') {
            // 去掉前导空格
            idx++;
        }
        if (idx == n) {
            //去掉前导空格以后到了末尾了
            return 0;
        }
        boolean negative = false;
        if (chars[idx] == '-') {
            //遇到负号
            negative = true;
            idx++;
        } else if (chars[idx] == '+') {
            // 遇到正号
            idx++;
        } else if (!Character.isDigit(chars[idx])) {
            // 其他符号
            return 0;
        }
        int ans = 0;
        while (idx < n && Character.isDigit(chars[idx])) {
            int digit = chars[idx] - '0';
            if (ans > (Integer.MAX_VALUE - digit) / 10) {
                // 本来应该是 ans * 10 + digit > Integer.MAX_VALUE
                // 但是 *10 和 + digit 都有可能越界，所有都移动到右边去就可以了。
                return negative? Integer.MIN_VALUE : Integer.MAX_VALUE;
            }
            ans = ans * 10 + digit;
            idx++;
        }
        return negative? -ans : ans;
    }
}
```



#### leetcode 842 将数组拆分成斐波那契序列

```java
class Solution{
  List<Integer> ans = new ArrayList<>();
  public List<Integer> splitIntoFibonacci(String S){
    return dfs(0,S,0,0,0)?ans:new ArrayList<>();
  }
  public boolean dfs(int p,String s,int pre1,int pre2,int deep){
    int length = s.length();
    if(p == length){
      return deep >= 3;
    }
    for(int i = 1;i <= 10;i++){
      if(p + i >length ||(s.charAt(p) == '0' && i > 1)){
        break;
      }
      String sub = s.substring(p,p+i);
      long numL = Long.parseLong(sub);
      if(numL > Integer.MAX_VALUE ||(deep != 0 && deep != 1 && numL > (pre1+pre2))){
        break;
      }
      Integer num = (int) numL;
      if(deep == 0||deep == 1||num.equals(pre1+pre2)){
        ans.add(num);
        if(dfs(p+i,s,pre2,num,deep+1)){
          return true;
        }
        ans.remove(num);
      }
    }
    return false;
  }
}
```



#### leetcode 509 斐波那契数

```java
class Solution {
    public int fib(int N) {
        if(N==0)
        	return 0;		
        int[] dp=new int[N+1];
		dp[0]=0;  
		dp[1]=1;
		for(int i=2;i<=N;i++) {
			dp[i]=dp[i-1]+dp[i-2];
		}
		return dp[N];
    }
}
```



#### leetcode 41 缺失的第一个正数

```java
class Solution{
  public int firstMissingPositive(int[] nums){
    int len = nums.length;
    
    for(int i = 0;i<len;i++){
      while(nums[i]>0 && nums[i]<=len&&nums[nums[i]-1]!=nums[i]){
        swap(nums,nums[i]-1,i);
      }
    }
    
    for(int i = 0;i<len;i++){
      if(nums[i]!=i+1){
        return i+1;
      }
    }
    return len+1;
  }
  
      private void swap(int[] nums, int index1, int index2) {
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }

}
```



#### 剑指offer 3原地置换

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        int temp;
        for(int i=0;i<nums.length;i++){
            while (nums[i]!=i){
                if(nums[i]==nums[nums[i]]){
                    return nums[i];
                }
                temp=nums[i];
                nums[i]=nums[temp];
                nums[temp]=temp;
            }
        }
        return -1;
    }
}
```



#### leetcode 704 二分查找

```java
class Solution {
  public int search(int[] nums, int target) {
    int pivot, left = 0, right = nums.length - 1;
    while (left <= right) {
      pivot = left + (right - left) / 2;
      if (nums[pivot] == target) return pivot;
      if (target < nums[pivot]) right = pivot - 1;
      else left = pivot + 1;
    }
    return -1;
  }
}
```



#### leetcode 128 最长连续序列

```java
	public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int max = 0;
        for (int num : nums) {
            if (!set.contains(num - 1)) {//判断set不包含当前元素-1的值，跳过已经计算的最长递增序列
                int curNum = num;
                int curCnt = 1;
                while (set.contains(curNum + 1)) {
                    curNum += 1;
                    curCnt += 1;
                }
                max = Math.max(max,curCnt);
            }
        }
        return max;
    }
```



#### leetcode 300 最长上升子序列

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        int ret = 0;
        for (int i = 0; i < n; i++) {
            int max = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    max = Math.max(max, dp[j] + 1);
                }
            }
            dp[i]=max;
            ret=Math.max(ret,dp[i]);
        }
        return ret;
    }
}

class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp,1);
        int ret = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            ret=Math.max(ret,dp[i]);
        }
        return ret;
    }
}
```



#### leetcode 673  最长递增子序列的个数

```java
public int findNumberOfLIS(int[] nums){
  if(nums == null || nums.length == 0){
    return 0;
  }
  int n = nums.length;
  int[] dp = new int[n];
  int[] counter = new int[n];
  Arrays.fill(dp,1);
  Arrays.fill(counter,1);
  int max = 0;
  for(int i = 0;i < n;i++){
    for(int j = 0;j < i;j++){
      if(nums[i] > nums[j]){
        if(dp[j] + 1 > dp[i]){
        	dp[i] = Math.max(dp[i],dp[j] + 1);
        	counter[i] = counter[j];
      	}else if(dp[j] + 1 == dp[i]){
        	counter[i] += counter[j];
      	}
      }
    }
    max = Math.max(max,dp[i]);
  }
  int res = 0;
  for(int i = 0;i < n;i++){
    if(dp[i] == max){
      res += counter[i];
    }
  }
  return res;
}
```



#### leetcode 814 二叉树剪枝

```java
public TreeNode pruneTree(TreeNode root) {
        if (root==null) return null;
        root.left=pruneTree(root.left);
        root.right=pruneTree(root.right);
        return (root.val==0&&root.left==null&&root.right==null)?null:root;
    }
```



#### 剑指offer 33 二叉搜索树的后序遍历

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        if(postorder == null||postorder.length == 0){
            return true;
        }
        int m = postorder.length;
        return helper(postorder,0,m-1);
    }

    public boolean helper(int[] p,int start,int end){
        if(end <= start){
            return true;
        }
        int i = start;
        for(;i < end;i++){
            if(p[i] > p[end]){
                break;
            }
        }
        for(int j = i;j < end;j++){
            if(p[j] < p[end]){
                return false;
            }
        }
        return helper(p,start,i - 1) && helper(p,i, end - 1);
    }
}
```



#### leetcode 23 合并K个链表

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        Queue<ListNode> pq = new PriorityQueue<>((v1, v2) -> v1.val - v2.val);
        for (ListNode node: lists) {
            if (node != null) {
                pq.offer(node);
            }
        }

        ListNode dummyHead = new ListNode(0);
        ListNode tail = dummyHead;
        while (!pq.isEmpty()) {
            ListNode minNode = pq.poll();
            tail.next = minNode;
            tail = minNode;
            if (minNode.next != null) {
                pq.offer(minNode.next);
            }
        }

        return dummyHead.next;
    }
}
```

 

#### leetcode 1 两数之和

```java
    public int[] twoSum(int[] nums,int target){
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int i = 0;i < nums.length;i++){
            int com = target - nums[i];
            if(map.containsKey(com)){
            return new int[]{map.get(com),i};
            }else{
            map.put(nums[i],i);
            }
        }
        return null;
    }
}
```



#### leetcode 445 两数之和II

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) { 
        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();
        while (l1 != null) {
            stack1.push(l1.val);
            l1 = l1.next;
        }
        while (l2 != null) {
            stack2.push(l2.val);
            l2 = l2.next;
        }
        
        int carry = 0;
        ListNode head = null;
        while (!stack1.isEmpty() || !stack2.isEmpty() || carry > 0) {
            int sum = carry;
            sum += stack1.isEmpty()? 0: stack1.pop();
            sum += stack2.isEmpty()? 0: stack2.pop();
            ListNode node = new ListNode(sum % 10);
            node.next = head;
            head = node;
            carry = sum / 10;
        }
        return head;
    }
}
```



#### leetcode 62 不同路径

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[] dp=new int[n];
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(j==0){
                    dp[j]=1;
                }else if(i==0){
                    dp[j]=1;
                }else{
                    dp[j]=dp[j-1]+dp[j];
                }
            }
        }
        return dp[n-1];
    }
}
```



#### leetcode 63 不同路径II

```java
 class Solution{   
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        if(obstacleGrid[0][0]==1||obstacleGrid[m-1][n-1]==1){
            return 0;
        }
        if(obstacleGrid[0][0]==0&&m==1&&n==1){
            return 1;
        }
        int[][] dp = new int[m][n];
        for(int i=0;i<m;i++){
            if(obstacleGrid[i][0]!=1){
                dp[i][0]=1;
            }else{
                break;
            }
        }
        for(int j=0;j<n;j++){
            if(obstacleGrid[0][j]!=1){
                dp[0][j]=1;
            }else{
                break;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] != 1) {
                    dp[i][j] = dp[i][j - 1] + dp[i-1][j];
                }
            }
        }
        return dp[m-1][n - 1];
    }
 }
```



#### leetcode 4 



#### 剑指offer 29 顺时针打印矩阵

```java
class Solution {
    public int[] spiralOrder(int[][] matrix) {
        ArrayList<Integer> list = new ArrayList<>();
        if(matrix == null||matrix.length == 0||matrix[0].length == 0){
            return new int[]{};
        }
        int m = matrix.length;
        int n = matrix[0].length;
        int[] res = new int[m * n]; 
        int top = 0;
        int bottom = m - 1;
        int left = 0;
        int right = n - 1;
        while(top <= bottom && left <= right){
            for(int i = left;i <= right;i++){
                list.add(matrix[top][i]);
            }
            top++;
            for(int i = top;i <= bottom;i++){
                list.add(matrix[i][right]);
            }
            right--;
            for(int i = right;i >= left && top <= bottom;i--){
                list.add(matrix[bottom][i]);
            }
            bottom--;
            for(int i = bottom;i >= top && left <= right; i--){
                list.add(matrix[i][left]);
            }
            left++;
        }
        int len = list.size();
        for(int i = 0;i < len;i++){
            res[i] = list.get(i);
        }
        return res;
    }
}
```



#### leetcode 509 斐波那契数

```java
class Solution {
    public int fib(int N) {
        if(N==0)
        	return 0;		
        int[] dp=new int[N+1];
		dp[0]=0;  
		dp[1]=1;
		for(int i=2;i<=N;i++) {
			dp[i]=dp[i-1]+dp[i-2];
		}
		return dp[N];
    }
}
```



#### leetcode 103 二叉树的锯齿形层次遍历

```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Deque<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int depth = 0;
        while (!queue.isEmpty()) {
            List<Integer> tmp = new LinkedList<>();
            int cnt = queue.size();
            for (int i = 0; i < cnt; i++) {
                TreeNode node = queue.poll();
                // System.out.println(node.val);
                if (depth % 2 == 0) tmp.add(node.val);
                else tmp.add(0, node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            res.add(tmp);
            depth++;
        }
        return res;
    }
}
```



#### leetcode 141 环形链表

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast)
                return true;
        }
        return false;
    }
}
```



#### 剑指offer 61 扑克牌中的顺子

```java
class Solution {
    public boolean isStraight(int[] nums) {
        int len = nums.length;
        if(len!=5) return false;
        Arrays.sort(nums);
        int numZeros=0;
        int interval=0;
        for(int i=0;i<len-1;i++){
            if(nums[i]==0){
                numZeros++;
                continue;
            }
            if(nums[i]==nums[i+1]){
                return false;
            }
            interval+=nums[i+1]-nums[i]-1;
        }
        if(numZeros>=interval){
            return true;
        }
        return false;
    }
}
```



#### leetcode 105 从前序与中序遍历序列构造二叉树

```java
class Solution {
	public TreeNode buildTree(int[] preorder, int[] inorder) {
		if(preorder.length==0 || inorder.length==0) {
			return null;
		}
		//根据前序数组的第一个元素，就可以确定根节点
		TreeNode root = new TreeNode(preorder[0]);
		for(int i=0;i<preorder.length;++i) {
			//用preorder[0]去中序数组中查找对应的元素
			if(preorder[0]==inorder[i]) {
				//将前序数组分成左右两半，再将中序数组分成左右两半
				//之后递归的处理前序数组的左边部分和中序数组的左边部分
				//递归处理前序数组右边部分和中序数组右边部分
				int[] pre_left = Arrays.copyOfRange(preorder,1,i+1);
				int[] pre_right = Arrays.copyOfRange(preorder,i+1,preorder.length);
				int[] in_left = Arrays.copyOfRange(inorder,0,i);
				int[] in_right = Arrays.copyOfRange(inorder,i+1,inorder.length);
				root.left = buildTree(pre_left,in_left);
				root.right = buildTree(pre_right,in_right);
				break;
			}
		}
		return root;
	}
}
```



#### leetcode 215 数组中的第K个最大元素

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        // init heap 'the smallest element first'
        PriorityQueue<Integer> heap =
            new PriorityQueue<Integer>();

        // keep k largest elements in the heap
        for (int n: nums) {
          heap.add(n);
          if (heap.size() > k)
            heap.poll();
        }

        // output
        return heap.poll();        
  }
}
```



#### leetcode 111 二叉树的最小深度

```java
class Solution{
  public int minDepth(TreeNode root){
    if(root == null){
      return 0;
    }
    int l = minDepth(root.left);
    int r = minDepth(root.right);
    if(l == 0 || r == 0){
      return l + r + 1;
    }
    return Math.min(l,r) + 1;
  }
}
```



#### 剑指offer 54  二叉搜索树的第K大结点

```java
class Solution {
    public int kthLargest(TreeNode root, int k) {
        // 在中序遍历的同时，把值加入表中
        ArrayList<Integer> list = new ArrayList<>();
        r(root,list);

        //话说倒数第k个数，下标是多少来着？诶，倒数第一个数下标是size-1诶，那么倒数第k个数不就是
        return list.get(list.size() - k);
    }

    // 二叉树递归形式中序遍历
    void r(TreeNode root, List list){
        if(root == null) return ;
        r(root.left,list);
        list.add(root.val);
        r(root.right,list);
    }
}
```



#### leetcode 994 腐烂的橘子

```java
public int orangesRotting(int[][] grid) {
    int M = grid.length;
    int N = grid[0].length;
    Queue<int[]> queue = new LinkedList<>();

    int count = 0; // count 表示新鲜橘子的数量
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            if (grid[r][c] == 1) {
                count++;
            } else if (grid[r][c] == 2) {
                queue.add(new int[]{r, c});
            }
        }
    }

    int round = 0; // round 表示腐烂的轮数，或者分钟数
    while (count > 0 && !queue.isEmpty()) {
        round++;
        int n = queue.size();
        for (int i = 0; i < n; i++) {
            int[] orange = queue.poll();
            int r = orange[0];
            int c = orange[1];
            if (r-1 >= 0 && grid[r-1][c] == 1) {
                grid[r-1][c] = 2;
                count--;
                queue.add(new int[]{r-1, c});
            }
            if (r+1 < M && grid[r+1][c] == 1) {
                grid[r+1][c] = 2;
                count--;
                queue.add(new int[]{r+1, c});
            }
            if (c-1 >= 0 && grid[r][c-1] == 1) {
                grid[r][c-1] = 2;
                count--;
                queue.add(new int[]{r, c-1});
            }
            if (c+1 < N && grid[r][c+1] == 1) {
                grid[r][c+1] = 2;
                count--;
                queue.add(new int[]{r, c+1});
            }
        }
    }

    if (count > 0) {
        return -1;
    } else {
        return round;
    }
}
```



#### leetcode 33 搜索旋转排序数组 

```java
public int search(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1, mid = 0;
    while (lo <= hi) {
        mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) {
            return mid;
        }
        // 先根据 nums[mid] 与 nums[lo] 的关系判断 mid 是在左段还是右段 
        if (nums[mid] >= nums[lo]) {
            // 再判断 target 是在 mid 的左边还是右边，从而调整左右边界 lo 和 hi
            if (target >= nums[lo] && target < nums[mid]) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        } else {
            if (target > nums[mid] && target <= nums[hi]) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
    }
    return -1;
}
```





#### leetcode 977 有序数组的平方

```java
class Solution {
    public int[] sortedSquares(int[] A) {
        // 新数组
        int[] nums = new int[A.length];
        for (int left = 0, right = A.length - 1, index = right; left <= right; ) {
            int powLeft = A[left] * A[left];
            int powRight = A[right] * A[right];
            // 从两边开始比较，比较的时候，只动值大的指针
            // 每次都插入值大的进数组
            if (powLeft > powRight) {
                left++;
                nums[index] = powLeft;
            } else {
                right--;
                nums[index] = powRight;
            }
            index--;
        }
        return nums;
    }
}
```



#### leetcode 344 反转字符串

```java
class Solution {
    public void reverseString(char[] s) {
        int i=0,j=s.length-1;
        while(i<j){
            char c=s[i];
            s[i]=s[j];
            s[j]=c;
            i++;
            j--;
        }
    }
}
```



#### leetcode 234 回文链表

```java
class Solution {

    public boolean isPalindrome(ListNode head) {

        if (head == null) return true;

        // Find the end of first half and reverse second half.
        ListNode firstHalfEnd = endOfFirstHalf(head);
        ListNode secondHalfStart = reverseList(firstHalfEnd.next);

        // Check whether or not there is a palindrome.
        ListNode p1 = head;
        ListNode p2 = secondHalfStart;
        boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) result = false;
            p1 = p1.next;
            p2 = p2.next;
        }        

        // Restore the list and return the result.
        firstHalfEnd.next = reverseList(secondHalfStart);
        return result;
    }

    // Taken from https://leetcode.com/problems/reverse-linked-list/solution/
    private ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode nextTemp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextTemp;
        }
        return prev;
    }

    private ListNode endOfFirstHalf(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
```



#### leetcode 1414 和为k的最少斐波那契数字数目

```java
class Solution {
    public int findMinFibonacciNumbers(int k) {
        int a = 1,b = 1;
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(1);
        while(a + b <= k){
            list.add(a + b);
            int temp = a + b;
            a = b;
            b = temp;
        }
        int ans = 0;
        for(int i = list.size()-1;i >=0;i--){
            if(k>=list.get(i)){
                ans++;
                k-=list.get(i);
            }
        }
        return ans;
    }
}
```



#### leetcode 324 摆动排序

```java
//快速选择排序
public int findKthLargest(int[] nums, int k) {
        if (nums.length == 0 || nums == null) return 0;
        int left = 0, right = nums.length - 1;
        while (true) {
            int position = partition(nums, left, right);
            if (position == k - 1) return nums[position]; //每一轮返回当前pivot的最终位置，它的位置就是第几大的，如果刚好是第K大的数
            else if (position > k - 1) right = position - 1; //二分的思想
            else left = position + 1;
        }
    }
 
    private int partition(int[] nums, int left, int right) {
        int pivot = left;
        int l = left + 1; //记住这里l是left + 1
        int r = right;
        while (l <= r) {
            while (l <= r && nums[l] >= nums[pivot]) l++; //从左边找到第一个小于nums[pivot]的数
            while (l <= r && nums[r] <= nums[pivot]) r--; //从右边找到第一个大于nums[pivot]的数
            if (l <= r && nums[l] < nums[pivot] && nums[r] > nums[pivot]) {
                swap(nums, l++, r--);
            }
        }
        swap(nums, pivot, r); //交换pivot到它所属的最终位置，也就是在r的位置，因为此时r的左边都比r大，右边都比r小
        return r; //返回最终pivot的位置
    }
```



#### leetcode 787 K 站中转内最便宜的航班

有 n 个城市通过 m 个航班连接。每个航班都从城市 u 开始，以价格 w 抵达 v。

现在给定所有的城市和航班，以及出发城市 src 和目的地 dst，你的任务是找到从 src 到 dst 最多经过 k 站中转的最便宜的价格。 如果没有这样的路线，则输出 -1。

```java
class Solution {
    //map[i][j]表示i到j的费用
    int[][] map;
    int result = Integer.MAX_VALUE;
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        map = new int[n][n];
        for(int[] flight:flights){
            map[flight[0]][flight[1]] = flight[2];
        }
        //可以经转k次可以看作可以飞行k+1次
        return methodOne(n,src, dst, K+1, 0);
    }

    //使用广度优先遍历递归搜索
    public int methodOne(int n, int cur, int dst, int K, int fee){
        if(cur == dst) return fee;
        if(K==0) return -1; //可以飞行的次数为0
        for(int i=0; i<n; i++){
            //fee+map[cur][i]<result可以进行剪枝
            if(map[cur][i]!=0 && fee+map[cur][i]<result){
                int temp = methodOne(n, i, dst, K-1, fee+map[cur][i]);
                if(temp!=-1) result = Math.min(result, temp);
            }
        }
        return result==Integer.MAX_VALUE?-1:result;
    }
}
```



#### leetcode 451 根据字符串出现频率排序

```java
class Solution {
    public String frequencySort(String s) {
        int[] letters = new int[128];
        for (char c : s.toCharArray()) {
            letters[c]++;
        }
        PriorityQueue<Character> heap = new PriorityQueue<>((a, b) -> Integer.compare(letters[b], letters[a]));
        StringBuilder res = new StringBuilder();

        for (int i = 0; i < letters.length; ++i) {
            if (letters[i] != 0) heap.offer((char)i);
        }

        while (!heap.isEmpty()) {
            char c = heap.poll();
            while (letters[c]-- > 0)
                res.append(c);
        }
        return res.toString();
    }
}

class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String input = sc.nextLine();
        char[] chars = input.toCharArray();
        HashMap<Character,Integer> map = new HashMap<>();
        for(char c:chars){
            map.put(c,map.getOrDefault(c,0)+1);
        }
        ArrayList<Map.Entry<Character,Integer>> list = new ArrayList<>();
        list.addAll(map.entrySet());
        Collections.sort(list,(m1,m2)->m2.getValue()-m1.getValue()==0?m1.getKey()-m2.getKey()
                :m2.getValue()-m1.getValue());
        StringBuilder sb = new StringBuilder();
        for(int i = 0;i<list.size();i++){
            int k = list.get(i).getValue();
            while(k-->0){
                sb.append(list.get(i).getKey());
            }
        }
        System.out.println(sb.toString());
    }
}

核心
   private static class ValueComparator implements Comparator<Map.Entry<Character, Integer>> {
        @Override
        public int compare(Map.Entry<Character, Integer> m, Map.Entry<Character, Integer> n) {
            //首先是按照 value 值来进行比较
            int v = n.getValue() - m.getValue();
            if (v == 0) {
                // 如果value值相同则按照key的值进行比较
                // 如果都是大写字母或小写字母按照字典顺序
                if ((!Character.isUpperCase(m.getKey()) && !Character.isUpperCase(n.getKey())) ||
                        (Character.isUpperCase(m.getKey()) && Character.isUpperCase(n.getKey()))) {
                    return m.getKey() - n.getKey();
                } else {
                    //如果有大写和小写，则小写在前，大写在后
                    return n.getKey() - m.getKey();
                }
            } else {
                return v;
            }
        }
    }
```



#### leetcode 236 二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode cur, TreeNode p, TreeNode q) {
        if (cur == null || cur == p || cur == q)
            return cur;
        TreeNode left = lowestCommonAncestor(cur.left, p, q);
        TreeNode right = lowestCommonAncestor(cur.right, p, q);
        //如果left为空，说明这两个节点在cur结点的右子树上，我们只需要返回右子树查找的结果即可
        if (left == null)
            return right;
        //同上
        if (right == null)
            return left;
        //如果left和right都不为空，说明这两个节点一个在cur的左子树上一个在cur的右子树上，
        //我们只需要返回cur结点即可。
        return cur;
    }
}

//二叉搜索树
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        if(root.val > p.val && root.val > q.val)
            return lowestCommonAncestor(root.left, p, q);
        return root;
    }
}
```



#### 剑指offer  51 逆序对个数

```java
public int reversePairs(int[] nums){
        int len = nums.length;
        
        if(len < 2){
            return 0;
        }
        
        int[] copy = new int[len];
        for(int i = 0; i < len;i++){
            copy[i] = nums[i];
        }
        int[] temp = new int[len];
        return reversePairs(copy,0,len - 1,temp);
    }
    
    private int reversePairs(int[] nums,int left,int right,int[] temp){
        if(left == right){
            return 0;
        }
        int mid = left + (right - left)/2;
        int leftPairs = reversePairs(nums,left,mid,temp);
        int rightPairs = reversePairs(nums,mid + 1,right,temp);
      	
      if(nums[mid] <= nums[mid + 1]){
        return leftPairs + rightPairs;
      }
        
        int crossPairs = mergeAndCount(nums,left,mid,right,temp);
        return leftPairs + rightPairs + crossPairs;
    }
    
    private int mergeAndCount(int[] nums,int left,int mid,int right,int[] temp){
        for(int i = left; i <=right;i++){
            temp[i]=nums[i];
        }
        int i = left;
        int j = mid + 1;
        
        int count = 0;
        for(int k = left;k <= right;k++){
            if(i == mid + 1){
                nums[k] = temp[j];
                j++;
            }else if(j == right + 1){
                nums[k] = temp[i];
                i++;
            }else if(temp[i] <= temp[j]){
                nums[k] = temp[i];
                i++;
            }else{
                nums[k] = temp[j];
                j++;
                count +=(mid - i + 1);
            }
        }
        return count;
    }
```



**归并排序**

```java
    void merge(int[] arr, int start, int end) {
        if (start == end) return;
        int mid = (start + end) / 2;
        merge(arr, start, mid);
        merge(arr, mid + 1, end);

        int[] temp = new int[end - start + 1];
        int i = start, j = mid + 1, k = 0;
        while(i <= mid && j <= end)
            temp[k++] = arr[i] < arr[j] ? arr[i++] : arr[j++];
        while(i <= mid)
            temp[k++] = arr[i++];
        while(j <= end)
            temp[k++] = arr[j++];
        System.arraycopy(temp, 0, arr, start, end);
    }

//归并解法 剑指offer51
    public int reversePairs(int[] nums) {
        return merge(nums, 0, nums.length - 1);
    }

    int merge(int[] arr, int start, int end) {
        if (start == end) return 0;
        int mid = (start + end) / 2;
        int count = merge(arr, start, mid) + merge(arr, mid + 1, end);

        int[] temp = new int[end - start + 1];
        int i = start, j = mid + 1, k = 0;
        while (i <= mid && j <= end) {
            count += arr[i] <= arr[j] ? j - (mid + 1) : 0;
            temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
        }
        while (i <= mid) {
            count += j - (mid + 1);
            temp[k++] = arr[i++];
        }
        while (j <= end)
            temp[k++] = arr[j++];
        System.arraycopy(temp, 0, arr, start, end - start + 1);
        return count;
    }

```



#### 剑指offer 3 数组中重复的数字

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        int temp;
        for(int i=0;i<nums.length;i++){
            while (nums[i]!=i){
                if(nums[i]==nums[nums[i]]){
                    return nums[i];
                }
                temp=nums[i];
                nums[i]=nums[temp];
                nums[temp]=temp;
            }
        }
        return -1;
    }
}
```

