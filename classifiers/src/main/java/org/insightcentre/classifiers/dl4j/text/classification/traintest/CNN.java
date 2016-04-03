package org.insightcentre.classifiers.dl4j.text.classification.traintest;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.classifiers.dl4j.ext.Dl4j_ExtendedWVSerializer;
import org.insightcentre.classifiers.utils.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import joptsimple.OptionParser;
import joptsimple.OptionSet;

public class CNN {          


	public static String TRAIN_DATA_PATH = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/suggData/all.csv";
	public static String TEST_DATA_PATH = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/suggData/electronics.csv";

	///home/sapneg/git/acl2016experiments/nn_classifiers/src/main/resources/suggData/electronics

	//public static String WORD_VECTORS_PATH_GOOGLE = "/home/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin";
	//public static final String WORD_VECTORS_PATH_GLOVE = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/models/glove.6B.50d.txt";
	//public static String WORD_VECTORS_PATH_GLOVE = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/models/glove.6B.50d.txt";
	//public static final String WORD_VECTORS_PATH_COMPOSES = "/home/kartik/Downloads/Mac-Downloads/John/STS/Publications/"
	//	+ "EN-wform.w.5.cbow.neg10.400.subsmpl.txt";	
	//public static final String WORD_VECTORS_PATH_COMPOSES = "/Users/sapna/Downloads/Composes/"
	//	+ "EN-wform.w.5.cbow.neg10.400.subsmpl.txt";
	public static String WORD_VECTORS_PATH_COMPOSES = "/home/sapneg/wv/Composes/EN-wform.w.5.cbow.neg10.400.subsmpl.txt";
	public static String WORD_VECTORS_PATH_GLOVE_TWITTER = "/home/sapneg/wv/glove/twiiter/glove.twitter.27B.200d.txt";	
	public static String WORD_VECTORS_TO_BE_USED = null;

	private static void badOptions(OptionParser p, String message){
		System.err.println("Error: "  + message);
		try {
			p.printHelpOn(System.err);
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.exit(-1);
	}

	public static void main(String[] args) throws Exception {	// Option Parser
		final OptionParser op = new OptionParser() {{
			accepts("trd", "The CSV Data File containing the training data").withRequiredArg().ofType(String.class);
			accepts("ted", "The CSV Data File containing the test data").withRequiredArg().ofType(String.class);			
			accepts("wv", "Word embeddings").withRequiredArg().ofType(String.class);
			accepts("composes", "If using composes, use this as parameter like -composes, otherwise don't use it.");
			accepts("ne", "Number of epochs").withRequiredArg().ofType(Integer.class);
			accepts("in", "Input layer size").withRequiredArg().ofType(Integer.class);
			accepts("hi", "Hidden layer size").withRequiredArg().ofType(Integer.class);            
			accepts("trb", "Train batch size").withRequiredArg().ofType(Integer.class);
			accepts("noOfFilters", "No of filters at first conv layer").withRequiredArg().ofType(Integer.class);
			accepts("typeOfFilter", "Type of filter (bigram, tri gram, etc.), e.g. 2 for bigram, and 3 for trigram ").withRequiredArg().ofType(Integer.class);			
			accepts("lr", "Learning rate").withRequiredArg().ofType(Double.class);
			accepts("ditClass", "Data Iterator class name with path to be used").withRequiredArg().ofType(String.class); 
			accepts("evalEveryN", "Evaluation every n epochs").withRequiredArg().ofType(Integer.class);
			accepts("modelToSavePath", "To save learnt model name").withRequiredArg().ofType(String.class);
			accepts("truncateLength", "Max tokens allowed i.e. truncate after this many tokens in the text").withRequiredArg().ofType(Integer.class);

		}};

		final OptionSet os;
		try {
			os = op.parse(args);
			int size = os.asMap().size();
			if(size!=15){
				op.printHelpOn(System.err);
			}
		} catch(Exception x) {
			badOptions(op, x.getMessage());
			return;
		}

		TRAIN_DATA_PATH = ((String)os.valueOf("trd"));
		TEST_DATA_PATH = ((String)os.valueOf("ted"));

		WORD_VECTORS_TO_BE_USED = ((String)os.valueOf("wv"));		

		int nEpochs = os.has("ne") ? (Integer) os.valueOf("ne") : 7;        //Number of epochs (full passes of training data) to train on

		int vectorSize = (Integer) os.valueOf("in");

		int hiddenLayerSize = os.has("hi") ? (Integer) os.valueOf("hi") : 10;

		int evalEveryN = os.has("evalEveryN") ? (Integer) os.valueOf("evalEveryN") : 2;

		int trainBatchSize = os.has("trb") ? (Integer) os.valueOf("trb") : 500;     //Number of examples in each minibatch

		int noOfFilters = os.has("noOfFilters") ? (Integer) os.valueOf("noOfFilters") : 20;     //Number of examples in each minibatch

		int typeOfFilter = os.has("typeOfFilter") ? (Integer) os.valueOf("typeOfFilter") : 2;     //Number of examples in each minibatch
		
		int truncateLength = os.has("truncateLength") ? (Integer) os.valueOf("truncateLength") : 40;
		
		double lr = os.has("lr") ? (Double) os.valueOf("lr") : 0.02;

		String className = (String) os.valueOf("ditClass");

		boolean composes = os.has("composes");
		
		String whereToSave = (String) os.valueOf("modelToSavePath");
		
		Class<? extends DataSetIterator> iteratorClass = (Class<? extends DataSetIterator>) Class.forName(className);

		//WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false);
		WordVectors wordVectors = null;
		//WordVectors wordVectors = Dl4j_ExtendedWVSerializer.loadTxtVectorsComposes(new File(WORD_VECTORS_PATH_COMPOSES));
		//WordVectors wordVectors = Dl4j_ExtendedWVSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH_GLOVE_TWITTER));

		if(composes){
			wordVectors = Dl4j_ExtendedWVSerializer.loadTxtVectorsComposes(new File(WORD_VECTORS_TO_BE_USED));
		} else {
			wordVectors = Dl4j_ExtendedWVSerializer.loadTxtVectors(new File(WORD_VECTORS_TO_BE_USED));
		}

		int testBatchSize = 500;
		int truncateReviewsToLength = truncateLength;  //Truncate reviews with length (# words) greater than this
						
		int seed = (int) System.nanoTime();
		int iterations = 1;
		int nChannels = 1;
		
		//log.info("Build model....");
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l2(0.0005)
				.learningRate(lr)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.list(4)
				.layer(0, new ConvolutionLayer.Builder(typeOfFilter, vectorSize)
						.nIn(nChannels)
						.stride(1, 1)
						.nOut(noOfFilters).dropOut(0.5)
						.activation("relu")
						.build())
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(truncateReviewsToLength-typeOfFilter, 1) //kernel size for performing max pooling over time as in the EMNLP paper CNN for sentence classification
						.stride(1,1)
						.build())
				.layer(2, new DenseLayer.Builder().activation("relu")
						.nOut(hiddenLayerSize).build())
				.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(2)
						.activation("softmax")
						.build())
				.backprop(true).pretrain(false);
		new ConvolutionLayerSetup(builder, truncateReviewsToLength, vectorSize, 1);

		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();		

		DataProducer ttiterator = new DataProducer(TRAIN_DATA_PATH, TEST_DATA_PATH, wordVectors, 
				trainBatchSize, testBatchSize, truncateReviewsToLength, iteratorClass);

		Pair<DataSetIterator, DataSetIterator> suggData = ttiterator.pair();
		DataSetIterator trainIt = suggData.getFirst();
		DataSetIterator testIt = suggData.getSecond();

		DataSetIterator train = new AsyncDataSetIterator(trainIt, 1);
		DataSetIterator test = new AsyncDataSetIterator(testIt,1);

		System.out.println("Starting training");

		for(int i=0; i<nEpochs; i++){
			net.fit(train);
			train.reset();

			System.out.println("Epoch " + i + " complete.");

			if(i%evalEveryN == 0){
				System.out.println("Starting evaluation:");

				Evaluation evaluation = new Evaluation();
				int testBatchCount = 0;
				while(test.hasNext()){
					DataSet t = test.next();
					INDArray features = t.getFeatureMatrix();
					INDArray labels = t.getLabels();
					INDArray output = net.output(features);
					evaluation.eval(labels, output);					
					System.out.print(testBatchCount++ + " ");
				}
				test.reset();
				System.out.println();
				System.out.println(evaluation.stats());

				printPRF(evaluation, 0, 1);
			}
		}

		SerializationUtils.saveObject(net, new File(whereToSave));
		
		System.out.println("");
		
		System.out.println("");
		System.out.println("");
		System.out.println("********************");
		
		System.out.println("All Epochs Complete, and model saved at: " +  whereToSave);

		
		System.out.println("");
		System.out.println("");
		System.out.println("");
		
		System.out.println("Starting final evaluation:");

		Evaluation evaluation = new Evaluation();
		int testBatchCount = 0;
		while(test.hasNext()){
			DataSet t = test.next();
			INDArray features = t.getFeatureMatrix();
			INDArray labels = t.getLabels();
			INDArray output = net.output(features);
			evaluation.eval(labels, output);					
			System.out.print(testBatchCount++ + " ");
		}
		test.reset();
		System.out.println();
		System.out.println(evaluation.stats());

		printPRF(evaluation, 0, 1);

		System.out.println();

		System.out.println("----- Evaluation complete -----");
	}

	public static void printPRF(Evaluation evaluation, int... classLabels){
		for(int classLabel : classLabels){
			System.out.println("Class " + classLabel +  ": ");
			System.out.print("P: " + evaluation.precision(classLabel) + " ");
			System.out.print("R: " + evaluation.recall(classLabel) + " ");
			System.out.println("F1: " + evaluation.f1(classLabel) + " ");
		}
	}

}