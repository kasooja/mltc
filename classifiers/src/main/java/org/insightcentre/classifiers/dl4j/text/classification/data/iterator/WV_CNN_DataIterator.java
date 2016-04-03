package org.insightcentre.classifiers.dl4j.text.classification.data.iterator;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.util.SerializationUtils;
import org.insightcentre.classifiers.utils.StanfordNLPUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import edu.stanford.nlp.ling.TaggedWord;

/** 
 * word vectors 
 * Labels/target: a single class (negative or positive), predicted at the final time step (word) of each review
 *
 * @author K
 */
public class WV_CNN_DataIterator implements DataSetIterator {

	private static final long serialVersionUID = 1L;
	private final WordVectors wordVectors;
	private final int batchSize;
	private final int vectorSize;
	private final int truncateLength;

	private int cursor = 0;

	private final List<String> positiveFiles;
	private final List<String> negativeFiles;

	private static Map<String, List<TaggedWord>> tagMap = new HashMap<String, List<TaggedWord>>();
	private static String tagMapPath = "src/main/resources/suggData/sentTag.map"; 

	static {
		if(new File(tagMapPath).exists()){
			tagMap = SerializationUtils.readObject(new File(tagMapPath));
		}
	}

	public WV_CNN_DataIterator(WordVectors wordVectors, List<String> positiveFiles, List<String> negativeFiles, int batchSize, int truncateLength){
		this.positiveFiles = positiveFiles;
		this.negativeFiles = negativeFiles;
		this.batchSize = batchSize;
		this.vectorSize = wordVectors.lookupTable().layerSize();
		this.wordVectors = wordVectors;
		this.truncateLength = truncateLength;
	}

	@Override
	public DataSet next(int num) {
		if (cursor >= positiveFiles.size() + negativeFiles.size()) throw new NoSuchElementException();
		try{
			return nextDataSet(num);
		}catch(IOException e){
			throw new RuntimeException(e);
		}
	}

	private DataSet nextDataSet(int num) throws IOException {
		//First: load reviews to String. Alternate positive and negative reviews
		List<String> reviews = new ArrayList<String>();
		boolean[] positive = new boolean[num];
		for( int i=0; i<num && cursor<totalExamples(); i++ ){
			if(cursor % 2 == 0){
				//Load positive review
				int posReviewNumber = cursor / 2;
				if(posReviewNumber<positiveFiles.size()){
					String review = positiveFiles.get(posReviewNumber).trim();
					reviews.add(review);
					positive[i] = true;
				}
			} else {
				//Load negative review
				int negReviewNumber = cursor / 2;		
				if(negReviewNumber<negativeFiles.size()){
					String review = negativeFiles.get(negReviewNumber).trim();
					reviews.add(review);
					positive[i] = false;
				}
			}
			cursor++;
		}

		//Second: tokenize reviews and filter out unknown words
		List<List<String>> allTokens = new ArrayList<List<String>>();
		int maxLength = 0;

		for(String s : reviews){
			if(!tagMap.containsKey(s)){
				List<TaggedWord> tagText = StanfordNLPUtil.getTagText(s);
				tagMap.put(s,  tagText);					
			}
			List<TaggedWord> tokens = tagMap.get(s);

			List<String> tokensFiltered = new ArrayList<>();
			for(TaggedWord t : tokens){
				if(wordVectors.hasWord(t.word().toLowerCase())) {
					tokensFiltered.add(t.word().toLowerCase().trim());
				}
			}
			allTokens.add(tokensFiltered);
			maxLength = Math.max(maxLength, tokensFiltered.size());
		}
		//If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
		//if(maxLength > truncateLength) 
		maxLength = truncateLength;

		//Create data for training
		//Here: we have reviews.size() examples of varying lengths
		//		INDArray features = Nd4j.create(reviews.size(), vectorSize, maxLength);
		INDArray features = Nd4j.create(reviews.size(), maxLength * vectorSize);

		//		INDArray feature = Nd4j.create(reviews.size(), maxLength);
		//		int rj = feature.rows();

		INDArray labels = Nd4j.create(reviews.size(), 2);    //Two labels: positive or negative
		//Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
		//Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
		//	INDArray featuresMask =
		//	Nd4j.zeros(reviews.size(), maxLength);
		//INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

		//int[] temp = new int[2];
		for( int i=0; i<reviews.size(); i++ ){
			List<String> tokens = allTokens.get(i);
			//temp[0] = i;
			INDArray overallVector = null;

			//Get word vectors for each word in review, and put them in the training data
			for( int j=0; j<tokens.size() && j<maxLength; j++ ){
				String token = tokens.get(j).trim();
				INDArray vector = wordVectors.getWordVectorMatrix(token);

				if(overallVector == null){
					overallVector = vector;
				} else {
					overallVector = Nd4j.concat(1, overallVector, vector);	
				}

				//		int rr = vector.rows();
				//	int cc = vector.columns();

				//temp[1] = j;
				//featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
			}
			if(overallVector==null){
				overallVector = Nd4j.zeros(maxLength * vectorSize);
			}			
			int columns = overallVector.columns();
			if(columns!=maxLength * vectorSize){
				int diff = (maxLength * vectorSize) - columns;
				INDArray zeros = Nd4j.zeros(diff);
				overallVector = Nd4j.concat(1, overallVector, zeros);				
			}
			features.putRow(i, overallVector);
		//	System.out.println();
			//int rows = features.rows();
			//int columns = features.columns();

			int idx = (positive[i] ? 0 : 1);

			//labels.putScalar(new int[]{i}, idx);   //Set label: [0,1] for negative, [1,0] for positive
			double[] label = null;

			if(idx == 0){
				label = new double[2];
				label[0] = 0;
				label[1] = 1;
			} else if(idx == 1) {
				label = new double[2];
				label[0] = 1;
				label[1] = 0;				
			}
			INDArray labInd = Nd4j.create(label);
			labels.putRow(i, labInd);
			//labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
		}

		return new DataSet(features,labels);
	}

	@Override
	public int totalExamples() {
		return positiveFiles.size() + negativeFiles.size();
	}

	@Override
	public int inputColumns() {
		return vectorSize;
	}

	@Override
	public int totalOutcomes() {
		return 2;
	}

	//	public void serialize(){
	//		SerializationUtils.saveObject(tagMap, new File(tagMapPath));
	//	}

	@Override
	public void reset() {		
		cursor = 0;
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException();
	}

	@Override
	public List<String> getLabels() {
		return Arrays.asList("positive","negative");
	}

	@Override
	public boolean hasNext() {
		return cursor < numExamples();
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

	@Override
	public void remove() {

	}


	public static void main(String[] args) {
		//WVPOS_SuggDataIterator d = new WVPOS_SuggDataIterator(null, null,  null,  4,  4);
		Constructor<?>[] constructors = WV_CNN_DataIterator.class.getConstructors();
		for(Constructor<?> constructor : constructors){
			Class<?>[] parameterTypes = constructor.getParameterTypes();
			for(Class<?> pa : parameterTypes){
				System.out.println(pa.getName());
			}
		}
	}
}