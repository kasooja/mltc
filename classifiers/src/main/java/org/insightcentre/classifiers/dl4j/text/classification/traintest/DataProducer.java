package org.insightcentre.classifiers.dl4j.text.classification.traintest;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.insightcentre.classifiers.utils.Pair;

import au.com.bytecode.opencsv.CSV;
import au.com.bytecode.opencsv.CSVReadProc;

public class DataProducer {

	private String trainDataPath = null; //.csv file
	private String testDataPath = null; //.csv file	
	private WordVectors wordVectors;
	private int trainBatchSize;
	private int testBatchSize;
	private int truncateReviewsToLength;
	private int csvTotCount = 0;
	private int invalidCsvRows = 0;
	private List<String> trainPos = new ArrayList<String>();
	private List<String> trainNeg = new ArrayList<String>();
	private List<String> testPos = new ArrayList<String>();
	private List<String> testNeg = new ArrayList<String>();
	private boolean train = true;

	private Constructor<? extends DataSetIterator> dataItConstructor;

	public DataProducer(String trainDataPath, String testDataPath, WordVectors wordVectors, int trainBatchSize,
			int testBatchSize, int truncateReviewsToLength, Class<? extends DataSetIterator> dataSetIterator) {
		this.trainDataPath = trainDataPath;
		this.testDataPath = testDataPath;		
		this.wordVectors = wordVectors;
		this.trainBatchSize = trainBatchSize;
		this.testBatchSize = testBatchSize;
		this.truncateReviewsToLength = truncateReviewsToLength;		
		dataItConstructor = getDataItConstructor(dataSetIterator);
		this.train = true;
		loadData(this.trainDataPath);
		this.train = false;
		loadData(this.testDataPath);
	}

	private Constructor<? extends DataSetIterator> getDataItConstructor(Class<? extends DataSetIterator> dataSetIterator){
		try {
			return dataSetIterator.getConstructor(WordVectors.class, List.class, List.class, int.class, int.class);
		} catch (NoSuchMethodException e) {
			e.printStackTrace();
		} catch (SecurityException e) {
			e.printStackTrace();
		}
		return null;
	}

	public void loadData(String dataPath){
		CSV csv = CSV.create();
		csvTotCount = 0;
		invalidCsvRows = 0;
		csv.read(dataPath, new CSVReadProc() {
			public void procRow(int rowIndex, String... values) {

				/*
				System.out.println(csvTotCount++);
				 */
				List<String> pos = null;
				List<String> neg = null;

				if(train){
					pos = trainPos;
					neg = trainNeg;
				} else if(!train){
					pos = testPos;
					neg = testNeg;
				}

				String id = values[0];
				if("id".equals(id)){
					return;
				}

				String text = values[1].trim();
				int suggLabel = Integer.parseInt(values[2].trim());

				if("text".equals(text) || "".equals(text) || (suggLabel != 0 && suggLabel != 1) || text.split("\\s+").length>50){
					invalidCsvRows++;
					return;
				}			

				if(suggLabel == 0){
					neg.add(text);					
				} else if(suggLabel == 1){
					pos.add(text);
				}

			}			
		});

		long seed = System.nanoTime();

		List<String> pos = null;
		List<String> neg = null;

		if(train){
			pos = trainPos;
			neg = trainNeg;
		} else if(!train){
			pos = testPos;
			neg = testNeg;
		}

		Collections.shuffle(neg, new Random(seed));
		Collections.shuffle(pos, new Random(seed));

		System.out.println("Total Rows Count: " +  csvTotCount);
		System.out.println("invalid Rows Count: " +  invalidCsvRows);
		System.out.println("Total Pos Examples: " + pos.size());
		System.out.println("Total Neg Examples: " + neg.size());		
	}

	public Pair<DataSetIterator, DataSetIterator> pair(){
		List<String> testNegs = new ArrayList<String>();
		List<String> testPoss = new ArrayList<String>();

		for(int i=0; i<testNeg.size(); i++){
			String negEx = testNeg.get(i);
			testNegs.add(negEx);	
		}

		for(int i=0; i<testPos.size(); i++){
			String posEx = testPos.get(i);
			testPoss.add(posEx);
		}

		DataSetIterator testIt = getNewDataItrInstance(testPoss, testNegs, testBatchSize);

		List<String> trainNegs = new ArrayList<String>();
		List<String> trainPoss = new ArrayList<String>();


		for(int i=0; i<trainNeg.size(); i++){
			String negEx = trainNeg.get(i);
			trainNegs.add(negEx);	
		}

		for(int i=0; i<trainPos.size(); i++){
			String posEx = trainPos.get(i);
			trainPoss.add(posEx);
		}

		Collections.shuffle(trainNegs);
		Collections.shuffle(trainPoss);

		if(trainNegs.size()==trainPoss.size()){
		} else if(trainNegs.size()<trainPoss.size()){
			int diff = trainPoss.size() - trainNegs.size();
			int origSize = trainNegs.size();
			for(int k=0; k<diff; k++){
				int[] randArray = new Random().ints(1, 0, origSize).toArray();
				int indexToBeAdded = randArray[0];
				trainNegs.add(trainNegs.get(indexToBeAdded));
			}
		} else if(trainPoss.size()<trainNegs.size()){
			int diff = trainNegs.size() - trainPoss.size();
			int origSize = trainPoss.size();
			for(int k=0; k<diff; k++){
				int[] randArray = new Random().ints(1, 0, origSize).toArray();
				int indexToBeAdded = randArray[0];
				trainPoss.add(trainPoss.get(indexToBeAdded));
			}
		}		

		System.out.println(trainPoss.size()==trainNegs.size());

		DataSetIterator trainIt = getNewDataItrInstance(trainPoss, trainNegs, trainBatchSize);	

		Pair<DataSetIterator, DataSetIterator> pair = new Pair<DataSetIterator, DataSetIterator>(trainIt, testIt);
		return pair;
	}

	private DataSetIterator getNewDataItrInstance(List<String> poss, List<String> negs, int batchSize){
		try {
			DataSetIterator newInstance = dataItConstructor.newInstance(wordVectors, poss, negs, batchSize, truncateReviewsToLength);
			return newInstance;
		} catch (InstantiationException e) {
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static void main(String[] args) {
		//		String DATA_PATH = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/suggData/all.csv";
		//		String WORD_VECTORS_PATH_GLOVE = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/models/glove.6B.50d.txt";
		//		WordVectors wordVectors = null;
		//		try {
		//			wordVectors = Dl4j_ExtendedWVSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH_GLOVE));
		//		} catch (FileNotFoundException e) {
		//			e.printStackTrace();
		//		}
		//		int trainBatchSize = 10;
		//		int testBatchSize = 100;
		//		int truncateReviewsToLength = 5;
	}

}
