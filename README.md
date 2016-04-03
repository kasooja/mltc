Text Classification Experiments
=======================================

1. Clone the project
2. Run "mvn install"
3. Enter the module folder by "cd classifiers"
4. Execute the particular java class using the following commands:

4a. To run Cross Fold using LSTM (class LSTM), see the example below:

MAVEN_OPTS=-Xmx16g mvn exec:java -Dexec.mainClass="org.insightcentre.classifiers.dl4j.text.classification.traintest.LSTM" -Dexec.args="-d src/main/resources/data/data.csv -wv src/main/resources/embeddings/Composes/EN-wform.w.5.cbow.neg10.400.subsmpl.txt -nf 10 -ne 10 -in 400 -hi 80 -trb 5 -lr 0.002 -ditClass org.insightcentre.classifiers.dl4j.text.classification.data.iterator.WV_DataIterator -composes"

4b. To run Train Test using CNN (class CNN), see the example below:

MAVEN_OPTS=-Xmx16g mvn exec:java -Dexec.mainClass="org.insightcentre.classifiers.dl4j.text.classification.traintest.CNN" -Dexec.args="-trd src/main/resources/data/training.csv -ted src/main/resources/test.csv -wv src/main/resources/embeddings/Composes/EN-wform.w.5.cbow.neg10.400.subsmpl.txt -ne 7 -in 400 -hi 100 -trb 10 -lr 0.004 -ditClass org.insightcentre.classifiers.dl4j.text.classification.data.iterator.WV_CNN_DataIterator -evalEveryN 2 -modelToSavePath src/main/resources/models/sampleCNN.model -composes -noOfFilters 50 -typeOfFilter 2"


Word Embeddings

For Testing:
	
		[Glove](http://nlp.stanford.edu/projects/glove/), specifically, Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip. [Download here](http://nlp.stanford.edu/data/glove.6B.zip).

For Evaluation (current plan):
		
		[Composes](http://clic.cimec.unitn.it/composes/semantic-vectors.html), specifically, Best predict vectors on this page (5-word context window, 10 negative samples, subsampling, 400 dimensions.) [Download here](http://clic.cimec.unitn.it/composes/materials/EN-wform.w.5.cbow.neg10.400.subsmpl.txt.gz).




Arguments for Train Test LSTM:

Option                  Description                                   
------                  -----------                                   
--composes              Composes word embeddings                      
--ditClass              Data Iterator class name with path to be used 
--evalEveryN <Integer>  Evaluation every n epochs                     
--hi <Integer>          Hidden layer size                             
--in <Integer>          Input layer size                              
--lr <Double>           Learning rate                                 
--modelToSavePath       To save learnt model name                     
--ne <Integer>          Number of epochs                              
--ted                   The CSV Data File containing the test data    
--trb <Integer>         Train batch size                              
--trd                   The CSV Data File containing the training data
--wv                    Word embeddings   
		




Arguments for Train Test CNN:

Option                      Description                                        
------                      -----------                                        
--composes                  If using composes, use this as parameter like -    
                              composes, otherwise don't use it.                
--ditClass                  Data Iterator class name with path to be used      
--evalEveryN <Integer>      Evaluation every n epochs                          
--hi <Integer>              Hidden layer size                                  
--in <Integer>              Input layer size                                   
--lr <Double>               Learning rate                                      
--modelToSavePath           To save learnt model name                          
--ne <Integer>              Number of epochs                                   
--noOfFilters <Integer>     No of filters at first conv layer                  
--ted                       The CSV Data File containing the test data         
--trb <Integer>             Train batch size                                   
--trd                       The CSV Data File containing the training data     
--truncateLength <Integer>  Max tokens allowed i.e. truncate after this many   
                              tokens in the text                               
--typeOfFilter <Integer>    Type of filter (bigram, tri gram, etc.), e.g. 2 for
                              bigram, and 3 for trigram                        
--wv                        Word embeddings       



Arguments for Train Test CNN-LSTM:

Option                      Description                                        
------                      -----------                                        
--composes                  If using composes, use this as parameter like -    
                              composes, otherwise don't use it.                
--ditClass                  Data Iterator class name with path to be used      
--evalEveryN <Integer>      Evaluation every n epochs                          
--hi <Integer>              Hidden layer size                                  
--in <Integer>              Input layer size                                   
--lr <Double>               Learning rate                                      
--modelToSavePath           To save learnt model name                          
--ne <Integer>              Number of epochs                                   
--noOfFilters <Integer>     No of filters at first conv layer                  
--ted                       The CSV Data File containing the test data         
--trb <Integer>             Train batch size                                   
--trd                       The CSV Data File containing the training data     
--truncateLength <Integer>  Max tokens allowed i.e. truncate after this many   
                              tokens in the text                               
--typeOfFilter <Integer>    Type of filter (bigram, tri gram, etc.), e.g. 2 for
                              bigram, and 3 for trigram                        
--wv                        Word embeddings