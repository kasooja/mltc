package org.insightcentre.classifiers.dl4j.text.classification.traintest;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
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

public class LSTM {          


	public static String TRAIN_DATA_PATH = "";
	public static String TEST_DATA_PATH = "";
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
			accepts("composes", "Composes word embeddings");
			accepts("ne", "Number of epochs").withRequiredArg().ofType(Integer.class);
			accepts("in", "Input layer size").withRequiredArg().ofType(Integer.class);
			accepts("hi", "Hidden layer size").withRequiredArg().ofType(Integer.class);            
			accepts("trb", "Train batch size").withRequiredArg().ofType(Integer.class);
			accepts("lr", "Learning rate").withRequiredArg().ofType(Double.class);
			accepts("ditClass", "Data Iterator class name with path to be used").withRequiredArg().ofType(String.class); 
			accepts("evalEveryN", "Evaluation every n epochs").withRequiredArg().ofType(Integer.class);
			accepts("modelToSavePath", "To save learnt model name").withRequiredArg().ofType(String.class);
		}};

		final OptionSet os;
		try {
			os = op.parse(args);
			int size = os.asMap().size();
			if(size!=12){
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
		int truncateReviewsToLength = 55;  //Truncate reviews with length (# words) greater than this

		//Set up network configuration
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.updater(Updater.RMSPROP)
				.regularization(true).l2(1e-5)
				.weightInit(WeightInit.XAVIER)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
				.learningRate(lr)
				.list(2)
				.layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(hiddenLayerSize)
						.activation("softsign").build())
				//.layer(1, new GravesLSTM.Builder().nIn(50).nOut(25)
				//		.activation("softsign").build())
				.layer(1, new RnnOutputLayer.Builder().activation("softmax")
						.lossFunction(LossFunctions.LossFunction.MCXENT).nIn(hiddenLayerSize).nOut(2).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		//net.setListeners(new ScoreIterationListener(10));		

		DataProducer ttiterator = new DataProducer(TRAIN_DATA_PATH, TEST_DATA_PATH, wordVectors, 
				trainBatchSize, testBatchSize, truncateReviewsToLength, iteratorClass);

		Pair<DataSetIterator, DataSetIterator> suggData = ttiterator.pair();
		DataSetIterator trainIt = suggData.getFirst();
		DataSetIterator testIt = suggData.getSecond();

		DataSetIterator train = new AsyncDataSetIterator(trainIt, 1);
		DataSetIterator test = new AsyncDataSetIterator(testIt,1);

		System.out.println("Starting training");

		for( int i=0; i<nEpochs; i++ ){
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
					INDArray inMask = t.getFeaturesMaskArray();
					INDArray outMask = t.getLabelsMaskArray();
					INDArray predicted = net.output(features, false, inMask, outMask);
					System.out.print(testBatchCount++ + " ");
					evaluation.evalTimeSeries(labels, predicted, outMask);
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
			INDArray inMask = t.getFeaturesMaskArray();
			INDArray outMask = t.getLabelsMaskArray();
			INDArray predicted = net.output(features, false, inMask, outMask);
			System.out.print(testBatchCount++ + " ");
			evaluation.evalTimeSeries(labels, predicted, outMask);
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