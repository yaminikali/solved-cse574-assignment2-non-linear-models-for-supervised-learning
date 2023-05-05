Download Link: https://assignmentchef.com/product/solved-cse574-assignment2-non-linear-models-for-supervised-learning
<br>
<strong>Note </strong>A zipped file containing skeleton Jupyter notebooks (PA2-Part1.ipynb and PA2-Part2.ipynb) and data is provided. Note that for each part, you need to write code in the specified function within the corresponding notebook file.

<strong>Evaluation </strong>For this assignment, we will evaluate your report. At the same time, we will also execute your submitted code to make sure that the findings documented in your report are observed in your code as well.

<strong>Submission </strong>You are required to submit a single file called <em>pa2.zip </em>using UBLearns. One submission per group is required. File <em>pa2.zip </em>must contain 3 files: <em>report.pdf</em>, <em>PA2-Part1.ipynb </em>and <em>PA2Part2.ipynb</em>.

<ul>

 <li>Submit your report in a pdf format. Please indicate the <strong>team members</strong>, <strong>group number</strong>, and your <strong>course number </strong>on the top of the report.</li>

 <li>The notebooks should contain all implemented functions. Please do not change the names of the files.</li>

</ul>

<strong>Please make sure that your group is enrolled in the UBLearns system</strong>: You should submit one solution per group through the groups page. <em>If you want to change the group, contact the instructors.</em>

<table width="88">

 <tbody>

  <tr>

   <td width="25"><strong>65</strong></td>

   <td width="63"><strong>Points</strong></td>

  </tr>

 </tbody>

</table>

<h1>Part I – Sentiment Analysis (Total)</h1>

In this part, you will classify movie reviews as positive or negative using the following two approaches and compare their accuracy.

<h2>Approach 1 – Word Count Vectorization</h2>

<strong>Task 1</strong>: Implement a simple hand-crafted classifier that can label a movie review as positive or negative.

<ul>

 <li>This part of the assignment is designed to help with the following:

  <ul>

   <li>Understanding the data and thus identifying the important features.</li>

   <li>Building a basic intuition of what model should be employed to solve a problem.</li>

  </ul></li>

 <li>Steps (<em>Some of these are already implemented</em>):

  <ol>

   <li>To get started with the exercise, you will need to download the supporting files and unzip its contents to the directory you want to complete this assignment in. Navigate to the root directory for the sentiment analysis dataset. Two files are provided: txt and labels.txt. Each line in reviews.txt correspond to a textual review for a movie. The corresponding line in the label files will have the label – NEGATIVE and POSITIVE, which is a binary sentiment inferred by reading the review. The files contain reviews and labels for 25000 different movies.</li>

   <li>Iterating through each review, obtain the count of each word, for positive and negative labels separately. These words can be referred to as the <em>vocabulary</em><a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. You will note that in the code, we are ignoring words that occur fewer than 100 times in the entire training data set (training corpus).</li>

  </ol></li>

</ul>

<table width="63">

 <tbody>

  <tr>

   <td width="18"><strong>10</strong></td>

   <td width="45"><strong>Points</strong></td>

  </tr>

 </tbody>

</table>

<ol start="3">

 <li>Calculate a feature for each word, based on the word counts for the positive and negative reviews, that can indicate the sentiment of the review. See the notebook for more details. []

  <ul>

   <li><em>Hint</em>: You can make use of the positive-to-negative ratio of a word from either category. Thus, a neutral word would have a ratio of around 1, as it appears just as often in both the categories.</li>

   <li>Convert the ratios into log values so that positive values are close to 1, negative values are close to -1 and neutral values are around 0.</li>

  </ul></li>

 <li>Use the word feature, defined above, to develop a simple “classifier”, that does not use a machine learning model. Instead, it should be a rule-based system that can assign a positive or negative label to each test review. Implement this logic in the nonml classifier function in the notebook.</li>

</ol>

[]

<strong>REPORT 1. </strong>

Describe your word features and the classification model and report the accuracy of your model on the test

data. []

<h2>Approach 2 – Neural Network Based Sentiment Classification</h2>

Using the previous approach, you are able to assign a sentiment to a review based on the positive or negative value of the words in a review. This is not a very sophisticated method to solve this particular problem but helps to build an intuition towards a more complex model we would use in Approach 2.

In the notebook, we have given you a code that allows you to create a single hidden layer, multi-layer perceptron (MLP), that is evaluated on the test review data.

<strong>REPORT 2. </strong>

<table width="48">

 <tbody>

  <tr>

   <td width="48"><strong>Points</strong></td>

  </tr>

 </tbody>

</table>

Report the accuracy and running time of the vanilla neural network. Is it better or worse than your rule-based algorithm implemented in Approach 1? [<strong>5</strong>]

<strong>Improving the Neural Network Performance</strong>

How can we make the neural network more accurate? In this part, we want you to explore different strategies. For each strategy you will have to report the final accuracy on the test data set and the running time.

<table width="63">

 <tbody>

  <tr>

   <td width="18"><strong>10</strong></td>

   <td width="45"><strong>Points</strong></td>

  </tr>

 </tbody>

</table>

<ul>

 <li>Making the model more complex: try increasing the “width” (number of units) in the hidden layer. Similarily, you can add more hidden layers to the architecture or increase the number of training epochs. []</li>

</ul>

<table width="63">

 <tbody>

  <tr>

   <td width="18"><strong>15</strong></td>

   <td width="45"><strong>Points</strong></td>

  </tr>

 </tbody>

</table>

<ul>

 <li>Modifying inputs to the neural network. In the first approach, you had developed word-level features. Could those be used to prune away some neutral words? Will that have an impact on the performance, in terms of accuracy and/or time? Implement the find ignore words function that identifies a set of words that can be ignored from the vocabulary. Re-run your code, including generation of the data file, and study the performance of the system. []</li>

</ul>

Figure 1: Sample sketches from the Quick Draw data set (Src: https://quickdraw.withgoogle.com/data)

<strong>REPORT 3. </strong>

Report the performance of the model (both in terms of accuracy and time) using different values for the hidden layer width, number of hidden layers, number of epochs and a set of ignored words. What has the most impact on the performance?

<h1>Part II – Image Classification on the AI Quick Draw Dataset (Total</h1>

<table width="88">

 <tbody>

  <tr>

   <td width="25"><strong>35</strong></td>

   <td width="63"><strong>Points</strong></td>

  </tr>

 </tbody>

</table>

<strong>)</strong>

The AI quick draw data set used for this part is created from the original Quick Draw dataset<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>. The original Quick Draw Dataset is a collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw! (See Figure 1).

The drawings were captured as timestamped vectors, tagged with metadata including what the player was asked to draw and in which country the player was located. However, for this assignment, we have provided you with a subsampled Quick Draw data set with the following properties:

<ul>

 <li>10 categories including ‘apple’, ‘basketball’, ‘arm’, ‘horse’, ‘ant’, ‘axe’, ‘alarm clock’, ‘airplane’, ‘banana’ and ‘bed’.</li>

 <li>Each category includes 12,500 images with the size of 28×28. Therefore, in the data set, each sample is a 784 length vector.</li>

 <li>The data set has already been separated as training features, training labels, testing features andtesting labels. In the code, we iteratively use the load function provided by the pickle package to extract those data from the data set.</li>

 <li><em>Note</em>: Since the data file is large, you will need to download it from the following link : https://www.</li>

</ul>

cse.buffalo.edu/ubds/docs/AI_quick_draw.pickle. This file is 94 MB in size. If, due to versioning issues, the pickle file is not readable on your system, you may use the provided create data set.py to recreate the pickle file.

<h2>Tasks</h2>

You will first study the performance of a multi-layered perceptron on the provided data as you change the complexity of the model by changing the number of hidden layers. You will then study the performance of the model when the fidelity (resolution) of the input image changes. Use the provided resize images() function to reduce the resolution of the image and then evaluate the performance (both accuracy and time) of the model on the test data set.

<strong>REPORT 4. </strong>

<table width="48">

 <tbody>

  <tr>

   <td width="48"><strong>Points</strong></td>

  </tr>

 </tbody>

</table>

<ol>

 <li>Run the evaluation of the 2 hidden layer neural network in the notebook – PA2-Part2.ipynb and report the test accuracy and the run time. [<strong>5</strong>]</li>

</ol>

<table width="63">

 <tbody>

  <tr>

   <td width="18"><strong>15</strong></td>

   <td width="45"><strong>Points</strong></td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>Compare the performance when the number of hidden layers are increased to 3 and 5. []</li>

 <li>Use the resize images() function to reduce the resolution of the images to (20 × 20), (15 × 15), (10 × 10) and (5 × 5). Using the 2 hidden layer architecture, compare the performance at different resolutions, including the original (28 × 28) resolution, both in terms of test accuracy and time. [<strong>15 Points</strong>]</li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> Hint: You can make use of the Counter function from collections library in python3.

<a href="#_ftnref2" name="_ftn2">[2]</a> https://quickdraw.withgoogle.com/data